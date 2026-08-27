// =============================================================================
// main2_final.cpp
// Integrasi Komprehensif: main2_upgraded + semua class performa dari main.cpp
// Tambahan:
//   - ThreadPool (extended: submitWithArgs, submitBatchAndWait, submitPriority)
//   - JobHandle + JobSystem (dependency graph, batch submit, FC6 style)
//   - TimelineSemaphore + TimelineSubmitInfo (Vulkan 1.2 timeline)
//   - VkStructs::AllocatedBuffer/AllocatedImage/DescriptorSetLayoutData
//   - SecondaryCommandBuffer + PerFrameThreadData (parallel CB recording)
//   - RenderResource + RenderGraph (automatic barrier/dependency resolution)
//   - FrustumCullingManager (extended: per-face DrawCall, merge batching)
//   - GPUObjectDescriptor + GPUCullPushConstants + IndirectDrawBuffer
//   - HaltonSequence (TAA jitter)
//   - GeometryGenerator (static class)
//   - ShaderModule (static class)
//   - VulkanMemoryAllocator (RAII allocation helper)
//   - FirstPersonCamera (quaternion-based)
//   - Semua diintegrasikan ke HelloTriangleApplication
// =============================================================================

// v74: SDL_MAIN_USE_CALLBACKS DICABUT.
//
// Makro itu memberi tahu SDL bahwa program memakai API callback
// (SDL_AppInit / SDL_AppIterate / SDL_AppEvent / SDL_AppQuit). Kode ini TIDAK
// memakainya — ia punya `int main()` dan loop sendiri. Di Termux makro itu
// tidak berakibat apa-apa karena SDL_main.h tidak ikut disertakan, jadi
// kesalahannya tidak pernah muncul.
//
// Di Android ia fatal: SDLActivity memanggil SDL_main lewat JNI, dan tanpa
// SDL_main.h nama `main` tidak pernah dipetakan ke situ. APK terpasang, lalu
// mati saat start.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>          // wajib untuk Android: main -> SDL_main
#include <SDL3/SDL_vulkan.h>

// v73: ukuran window sebenarnya, diisi App::initWindow() sesudah window jadi.
static uint32_t g_windowWidth  = 800;
static uint32_t g_windowHeight = 600;

// =============================================================================
//  v73 — PEMUATAN BERKAS PORTABEL (prasyarat APK Android)
//
//  MASALAHNYA. Di Android, berkas aplikasi tinggal di DALAM APK sebagai aset
//  terkompresi. Ia tidak punya path di filesystem, jadi std::ifstream tidak
//  akan pernah bisa membukanya — dan gagalnya sunyi: is_open() mengembalikan
//  false, shader kosong, lalu vkCreateShaderModule crash tanpa menyebut sebab.
//
//  SDL_LoadFile() menyelesaikannya: di Android ia membaca lewat AAssetManager
//  secara otomatis, di Linux/Termux ia membaca berkas biasa. SATU jalur untuk
//  keduanya, tanpa #ifdef, sehingga biner Termux dan APK memakai kode yang
//  persis sama.
//
//  Ketiga tempat yang dulu memakai ifstream sekarang lewat sini:
//    ShaderModule::ReadFile      (.spv)
//    ShadowAAA::Pipeline::readFile (.spv)
//    GeometryGenerator::loadObj  (tree.obj)
// =============================================================================
static std::vector<char> platformLoadFile(const std::string& path, bool* ok = nullptr) {
    size_t len = 0;
    void*  raw = SDL_LoadFile(path.c_str(), &len);
    if (!raw) {
        if (ok) *ok = false;
        return {};
    }
    std::vector<char> out(static_cast<const char*>(raw),
                          static_cast<const char*>(raw) + len);
    SDL_free(raw);
    if (ok) *ok = true;
    return out;
}

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/norm.hpp>

#include <vulkan/vulkan.hpp>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>   // std::atol, dipakai pemuat OBJ
#include <cfloat>    // FLT_MAX, dipakai AABB mesh
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <variant>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <cstdint>
#include <limits>
#include <float.h>

// === MULTITHREADING HEADERS ===
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <barrier>          // C++20
#include <latch>            // C++20
#include <semaphore>        // C++20
#include <span>             // C++20

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================
namespace Config {
    constexpr uint32_t WINDOW_WIDTH         = 800;
    constexpr uint32_t WINDOW_HEIGHT        = 600;

    // v73: ukuran yang BENAR-BENAR dipakai, diisi sesudah window dibuat.
    // Di Termux nilainya sama dengan konstanta di atas; di Android ia jadi
    // resolusi layar sebenarnya. Semua perhitungan aspek rasio memakai ini,
    // bukan konstantanya, supaya gambar tidak gepeng di layar HP.
    // ---- TAA (Temporal Anti-Aliasing) ------------------------------------
    // Menghapus gerigi dengan menggeser proyeksi kurang dari satu piksel tiap
    // frame lalu menumpuk hasilnya. Sekarang mungkin karena warna sudah mendarat
    // di target offscreen lebih dulu (Fase 3 langkah 1).
    //
    // TAA_ENABLED = false mematikannya: jitter jadi nol dan resolve cuma
    // meneruskan frame ini apa adanya.
    // Dinyalakan atas permintaanmu. Rantai jitter-nya sudah saya audit ulang dan
    // BENAR: prevViewProj dibangun dengan jitter frame INI (bukan frame lalu),
    // sehingga suku jitter saling menghapus saat pre-pass menghitung velocity —
    // motion vector-nya bebas jitter dan TAA tidak mengejar goyangannya sendiri.
    constexpr bool  TAA_ENABLED      = true;
    constexpr int   TAA_JITTER_COUNT = 8;      // panjang deret Halton
    constexpr float TAA_BLEND_MAX    = 0.92f;  // porsi history saat diam
    constexpr float TAA_SHARPNESS    = 0.35f;  // unsharp, 0 = mati

    // ---- Tonemap (Fase 3) ------------------------------------------------
    // Exposure mengalikan warna HDR sebelum kurva. Di bawah 1 menggelapkan,
    // di atas 1 menerangkan. ACES memberi bahu yang melandai di ujung terang
    // sehingga nilai jauh di atas 1.0 tetap punya beda yang terlihat.
    // TONEMAP_ACES = false mengembalikan pemotongan keras seperti sebelumnya.
    constexpr float TONEMAP_EXPOSURE        = 1.0f;
    constexpr bool  TONEMAP_ACES            = true;

    // ---- FXAA (v44) --------------------------------------------------------
    //
    // Terukur sebelum ini: transisi siluet 0 piksel (227 227 227 | 114 114 114).
    // TAA sudah nyala tapi m_taaHealth ~ 0 di 0,5 FPS — ia hidup dari frame
    // rate, jadi di llvmpipe praktis mati. FXAA bekerja per frame, tanpa
    // riwayat, jadi ia satu-satunya AA yang tetap jalan di sini.
    //
    // Implementasinya FXAA 3.11 preset kualitas 12 dari Timothy Lottes: deteksi
    // kontras lokal, penentuan arah tepi berbobot, penelusuran ujung tepi 12
    // langkah dua arah, plus koreksi subpiksel. Bukan versi ringkas 5-tap.
    constexpr bool  FXAA_ENABLED            = true;

    // Kekuatan koreksi subpiksel. 0 = hanya tepi panjang yang dihaluskan,
    // 1 = agresif dan gambar mulai terasa lembut. 0,75 nilai baku Lottes.
    // v45: 0,75 -> 1,00. Sliver 1-2 piksel (rusuk menara yang berkali-kali
    // dikira light bleeding) ditangani oleh JALUR SUBPIKSEL, bukan jalur
    // penelusuran tepi — fitur setipis itu tidak punya tepi panjang untuk
    // ditelusuri. Jadi knob inilah yang menentukan apakah ia hilang.
    constexpr float FXAA_SUBPIX             = 1.00f;

    // Ambang kontras relatif. 0,333 = murah/kasar, 0,063 = kualitas tertinggi.
    // v45: 0,125 -> 0,063, preset kualitas tertinggi Lottes. Lebih banyak
    // piksel membayar penelusuran, tapi tepi berkontras sedang ikut tertangkap.
    constexpr float FXAA_EDGE_THRESHOLD     = 0.063f;

    // Ambang kontras mutlak — di bawah ini piksel dianggap datar dan
    // dilewatkan tanpa biaya. Ini yang membuat FXAA murah: mayoritas layar
    // keluar lebih awal di sini, dan hanya piksel tepi yang membayar
    // penelusuran 12 langkahnya.
    constexpr float FXAA_EDGE_THRESHOLD_MIN = 0.0312f;

    // Visualisasi: 0 = normal, 1 = cat piksel tepi MAGENTA, 2 = besar geseran
    // sebagai gradasi hijau. Dipakai kalau FXAA terlihat tidak berpengaruh —
    // FXAA yang gagal tidak menghasilkan error, ia cuma diam.
    constexpr float FXAA_DEBUG              = 0.0f;

    // ---- LANGIT ANALITIK + BERKAS CAHAYA DI LANGIT (v36) -----------------
    //
    // Sampai build sebelumnya piksel langit tidak punya fragmen sama sekali:
    // ia berisi clear color (0,10 0,10 0,15) dari awal sampai akhir. Akibatnya
    // buffer volumetric — yang dihitung untuk SETIAP piksel — dibuang persis
    // di tempat nilainya paling besar. Piksel langit punya rayLen penuh
    // VOL_MAX_DIST, jadi inScatter-nya 0,5730 menghadap matahari dan 0,1002 di
    // 45 derajat, sementara langit yang menampungnya cuma 0,10.
    //
    // SKY_ENABLED = false mengembalikan perilaku lama persis (clear color).
    constexpr bool  SKY_ENABLED             = true;

    // Kekeruhan udara Preetham. 1,7 = vakum teoretis, 2..3 = langit cerah,
    // 6+ = berkabut/berpolusi. Turbidity tinggi memucatkan langit dan
    // MELEBARKAN aureole di sekitar matahari, jadi ia juga menaikkan kontras
    // pangkal berkas cahaya.
    constexpr float SKY_TURBIDITY           = 2.5f;

    // Luminansi zenit dalam satuan engine. Ini satu-satunya angka yang
    // menentukan kecerahan langit; sisanya bentuk yang dihitung model.
    // Terukur setelah ACES pada exposure 1,0 (matahari elevasi 31 derajat,
    // turbidity 2,5):
    //
    //   nilai   zenit di layar   horizon   berkas thd langit
    //    0,45   169 201 229       231       13,4 %
    //    0,60   188 215 237       238       10,0 %
    //    0,75   200 223 241       242        8,0 %
    //
    // Sebagai pembanding: lantai yang kena matahari penuh terbaca 186.
    // Makin terang langitnya, makin pucat horizonnya dan makin tenggelam
    // berkas cahayanya — 0,45 titik seimbangnya.
    constexpr float SKY_ZENITH_LUMA         = 0.45f;

    // Kecerahan piringan matahari. ACES sudah memetakan apa pun di atas ~8
    // menjadi putih, jadi angka ini menentukan seberapa jauh pijarnya menular
    // ke tetangga lewat tonemap, bukan warnanya sendiri.
    constexpr float SKY_SUN_INTENSITY       = 30.0f;

    // Pengali berkas cahaya KHUSUS langit. 1,0 = tepat sama dengan yang
    // diterima permukaan, jadi berkas yang sama tidak berubah kecerahan saat
    // menyeberangi siluet bangunan. Menaikkannya memutus kesinambungan itu —
    // untuk berkas yang lebih tegas, naikkan VOL_DENSITY supaya keduanya ikut.
    constexpr float SKY_SHAFT_SCALE         = 1.0f;

    constexpr uint32_t NUM_CASCADES         = 4; // KONSTANTA LAMA, tidak dipakai —
                                                 // yang berlaku ShadowAAA::Cfg::NUM_CASCADES

    // Shadow technique distances (dari main.cpp)

    // Camera settings
    constexpr float CAMERA_NEAR_PLANE       = 0.1f;
    // v41: 100 -> 160. Peta baru berdiagonal 128 m; dengan far plane 100 m,
    // sudut peta yang berlawanan tidak pernah digambar sama sekali.
    //
    // Ini TIDAK gratis. Split cascade membentang dari near sampai
    // min(farZ, fitDist), jadi memperjauhnya menipiskan texel di SEMUA cascade.
    // Terukur (6 cascade, RES 2048, lambda 0,88):
    //
    //   peta lama, far 100  -> jangkauan  34,0 m : C0 0,09  C3 0,60  C5  2,96 cm
    //   peta baru, far 100  -> jangkauan 100,0 m : C0 0,22  C3 1,44  C5  8,79 cm
    //   peta baru, far 160  -> jangkauan 152,7 m : C0 0,31  C3 2,03  C5 13,3  cm
    //
    // 160 dipilih karena di situ jangkauan cascade (152,7 m) baru melampaui
    // diagonal peta (128 m) — menaikkannya lagi cuma menipiskan texel untuk
    // ruang kosong. Kalau ketajaman lebih penting daripada jangkauan, turunkan
    // ke 100: seluruh peta tetap bisa dijelajahi, hanya sudut terjauh yang
    // ter-fog oleh far plane.
    constexpr float CAMERA_FAR_PLANE        = 160.0f;
    constexpr float CAMERA_FOV              = 45.0f;

    // Game settings
    constexpr float MAP_SCALE               = 1.5f;
    constexpr float PLAYER_RADIUS           = 0.2f;
    constexpr float PLAYER_HEIGHT           = 0.6f;
    constexpr float PLAYER_SPEED            = 5.0f;

    // Joystick settings
    constexpr float JOYSTICK_DEADZONE       = 0.1f;
    constexpr float JOYSTICK_RELATIVE_SIZE  = 0.15f;
    constexpr float JOYSTICK_KNOB_RATIO     = 0.4f;
    constexpr float JOYSTICK_MARGIN         = 40.0f;

    // Frustum culling
    constexpr float WALL_HEIGHT             = 1.5f;

    // ---- PETA BESAR (v41) --------------------------------------------------
    //
    // Peta lama 10x15 sel = 15 x 22,5 m: arena berdinding yang lebih kecil
    // daripada satu cascade. Cascade 3-5 (jangkauan 15-100 m) praktis tidak
    // pernah berisi apa pun, jadi separuh sistem CSM tidak pernah diuji.
    //
    // Peta baru 60x61 sel = 90 x 91,5 m, diagonal 128 m. Peta lama DISALIN
    // UTUH ke tengahnya, jadi titik spawn, pohon, tiang, dan bola tetap di
    // koordinat dunia yang sama persis — generator memusatkan peta di origin
    // (ox = -mW*ms/2), dan lebar/tinggi baru dipilih SEPARITAS dengan yang lama
    // (60 genap seperti 10, 61 ganjil seperti 15) supaya offset penyalinannya
    // bilangan bulat dan tidak ada pergeseran setengah sel.
    //
    // Nilai sel sekarang KODE TINGGI, bukan lagi 0/1: 1 = 1x WALL_HEIGHT,
    // 2 = 2x, dan seterusnya. Bangunan bertingkat inilah yang membuat cascade
    // jauh punya sesuatu untuk dibayangi.
    constexpr bool     MAP_BIG           = true;
    constexpr int      MAP_BIG_W         = 200;  // genap, separitas peta lama (10)
    constexpr int      MAP_BIG_H         = 201;  // ganjil, separitas peta lama (15)
    constexpr uint32_t MAP_SEED          = 1337u;
    constexpr int      MAP_MAX_LEVELS    = 22;   // kode tinggi tertinggi = 33 m
    constexpr int      MAP_BLOCK         = 7;    // periode kisi jalan, dalam sel
    constexpr float    MAP_FLOOR_MARGIN  = 20.0f; // lantai menjulur di luar peta, m

    // ---- ANGGARAN SEL (v42) ------------------------------------------------
    //
    // Pre-pass menggambar SELURUH level tanpa culling kamera
    // (drawIndexed(m_levelIndexCount) di drawScene dengan cascadeIndex < 0),
    // jadi jumlah sel terisi adalah biaya TETAP tiap frame — bukan jumlah draw
    // call, dan bukan yang terlihat di layar. Itu batasan yang mengikat di
    // llvmpipe, dan itu yang mengatur ketiga angka di bawah.
    //
    // Disimulasikan lebih dulu di sim_openworld.py sebelum ditulis ke sini:
    //
    //   inti kota    :  870 sel   (padat, rendah, bisa dijelajahi)
    //   cincin tengah:  577 sel   (bangunan sedang, mengisi jarak C3-C4)
    //   menara luar  : 1031 sel dalam 52 menara
    //   TOTAL        : 2478 sel   = 1,81x v41 (1372)
    //
    // Tidak ada dinding tepi peta: 800 sel untuk pagar yang hampir selalu di
    // luar far plane adalah biaya tetap yang tidak membeli apa-apa. Batas
    // visual dunia diserahkan ke cincin menara terluar.
    constexpr float    MAP_CORE_R        = 0.30f;  // radius inti, fraksi setengah-peta
    constexpr float    MAP_MID_R         = 0.62f;  // batas luar cincin menengah
    constexpr int      MAP_CORE_SKIP     = 68;     // % sel inti yang dikosongkan
    constexpr int      MAP_MID_SKIP      = 93;     // % sel cincin menengah yang dikosongkan
    constexpr int      MAP_TOWER_TRIES   = 4000;   // percobaan penempatan menara
    constexpr int      MAP_TOWER_SKIP    = 974;    // per-1000 percobaan yang ditolak

    // ---- KUBUS UJI BERDIRI SENDIRI ---------------------------------------
    // Objek pertama di engine ini yang TIDAK berasal dari levelMap. Gunanya
    // membuktikan seluruh jalur geometri -> shadow map tanpa perlu mesh rumit:
    // kalau bayangannya muncul dan memanjang mengikuti posisi matahari, maka
    // menambah pohon atau mesh glTF nanti tinggal soal jumlah segitiga.
    //
    // Ditaruh di (0, -3) supaya jatuh persis di depan spawn: pemain muncul di
    // (0, PLAYER_HEIGHT, 0) menghadap -Z, jadi kubusnya langsung terlihat tanpa
    // perlu berjalan. Sel peta yang ditumpanginya — (4,5) dan (5,5) — kosong,
    // begitu pula kedelapan tetangganya, jadi bayangannya jatuh di lantai
    // terbuka dan mudah diamati.
    //
    // Tingginya SENGAJA 1,0 m, beda dari WALL_HEIGHT 1,5 m. Kalau tingginya
    // sama, bayangan kubus dan bayangan dinding akan sulit dibedakan.
    // Tapak dan tinggi DIPISAH. Awalnya kubus 1x1x1 m, dan simulasi PCSS
    // menunjukkan bentuk itu tidak bisa membuktikan apa pun soal penumbra:
    //
    //   caster 1,0 m  -> penumbra PCSS mentah 0,27..1,57 cm
    //   tapi lantai layar MIN_PCF_SCREEN_PX * worldPerPixel = 1,64 cm
    //   -> lebar filter terpakai RATA 1,64 cm sepanjang bayangan (1,00x)
    //
    // Jadi seluruh keluaran PCSS tertelan lantai layar. Itu bukan cacat: dengan
    // SUN_ANGULAR_TAN 0,009 (matahari sungguhan ~0,0093), benda setinggi 1 m
    // memang hanya menghasilkan penumbra satu-dua sentimeter. Yang salah adalah
    // benda ujinya, bukan algoritmanya.
    //
    // Tiang 0,5 x 4,0 m membuat suku PCSS menang telak:
    //   0,49 cm -> 6,55 cm sepanjang bayangan 6,67 m
    //   lebar filter terpakai 1,64 -> 6,55 cm, pelebaran 3,99x
    //   di layar (jarak pandang 4 m): 6 px -> 24 px
    // Tapak sempit juga menguji geometri tipis, yang jadi kasus sesungguhnya
    // begitu batang pohon atau mesh glTF masuk.
    // CATATAN: indeks sekarang uint32. Batas 65 536 verteks yang dulu jadi
    // ranjau senyap sudah tidak ada — dulu (uint16_t)(vb+k) MEMBUNGKUS diam-diam
    // begitu level melewati batas itu, dan gejalanya bukan error melainkan
    // segitiga yang menunjuk verteks acak. Satu mesh glTF berdaun detail saja
    // cukup untuk memicunya.
    // ---- MESH OBJ DARI BERKAS -------------------------------------------
    // Kosongkan untuk mematikan. Kalau berkasnya tidak ada atau gagal diurai,
    // aplikasi tetap jalan dengan peringatan di log — mesh sekadar tidak muncul.
    //
    // OBJ dipilih lebih dulu daripada glTF dengan sengaja: ia teks murni tanpa
    // JSON, tanpa buffer biner, tanpa base64. Ia sudah cukup untuk menguji
    // seluruh jalur mesh yang selama ini belum pernah tersentuh — verteks
    // BERSAMA antar segitiga, normal halus, jumlah segitiga sembarang, dan
    // indeks di atas 65 536. Kalau ada asumsi kotak yang tersisa di pipeline,
    // di sinilah ia akan pecah, dan pecahnya murah.
    //
    // Transform dipanggang di CPU saat load. Itu bukan kompromi malas: push
    // constant pipeline pre-pass sudah terpakai penuh 128 byte, tepat di batas
    // minimum yang dijamin Vulkan, jadi matriks model per objek TIDAK MUAT di
    // sana. Jalur yang benar nanti adalah SSBO transform + indeks instance, dan
    // itu perubahan descriptor di tiga pipeline. Sampai saat itu, memanggang di
    // CPU memberi hasil yang identik untuk objek statis.
    // ---- BOLA MELAYANG YANG BERGERAK -------------------------------------
    // Objek DINAMIS pertama di engine ini. Semua geometri lain dipanggang
    // sekali saat startup; bola ini menulis ulang posisi verteksnya tiap frame
    // ke buffer yang terpetakan di memori host.
    //
    // Gunanya bukan hiasan. Setelah matahari diperlambat jadi orbit 6 jam,
    // tidak ada lagi yang bergerak di scene — dan penyaring temporal yang baru
    // dinaikkan ke TAU 40 detik jadi tidak pernah diuji terhadap perubahan.
    // Bola ini yang mengujinya: bayangannya menyapu lantai, dan kalau
    // penolakan riwayat di shadow_temporal bekerja, sapuan itu bersih tanpa
    // jejak. Kalau tidak, jejaknya akan panjang dan langsung terlihat.
    //
    // Ditaruh di samping tiang (tiang di x=0, z=-3) supaya kedua bayangan
    // terlihat bersamaan: yang satu diam, yang satu bergerak.
    constexpr bool  BALL_ENABLED     = true;
    constexpr float BALL_RADIUS      = 0.35f;
    constexpr float BALL_X           = 1.30f;   // di samping tiang
    constexpr float BALL_Y           = 1.10f;   // melayang di atas lantai
    constexpr float BALL_Z           = -3.00f;  // pusat lintasan
    constexpr float BALL_TRAVEL      = 1.60f;   // simpangan maju-mundur, meter
    // Satu siklus maju-mundur penuh. 40 detik dipilih dari frame rate, bukan
    // selera: pada 0,5 FPS itu 20 frame per siklus, jadi gerakannya terbaca
    // sebagai gerak. Nilai 12 detik yang sempat saya tulis cuma memberi 6
    // frame per siklus - bolanya akan tampak melompat antara enam posisi,
    // dan lompatan sebesar itu juga percuma sebagai uji penyaring temporal
    // karena tiap frame riwayatnya pasti ditolak.
    constexpr float BALL_PERIOD_SEC  = 40.0f;
    constexpr int   BALL_SEGMENTS    = 16;      // bujur
    constexpr int   BALL_RINGS       = 10;      // lintang

    constexpr const char* MESH_OBJ_PATH  = "tree.obj";
    constexpr float       MESH_SCALE     = 1.0f;

    // Ditaruh TEPAT DI DEPAN TIANG dilihat dari spawn: pemain muncul di
    // (0, PLAYER_HEIGHT, 0) menghadap -Z, tiang ada di z = -3,0, jadi batang
    // pohon di z = -1,95 berdiri satu meter di depannya.
    //
    // MESH_X = -0,12 bukan 0: kotak pembatas pohon condong ke kanan
    // (x -0,94..+1,19), jadi digeser setengah selisihnya supaya mahkotanya
    // terpusat di sumbu pandang, bukan batangnya.
    //
    // Terperiksa: kotak pembatas dunia x -1,06..1,07  z -2,71..-0,90; keempat
    // sel peta yang ditempati kosong; tidak menembus tiang (sisa celah 4 cm).
    //
    // CATATAN JARAK PANDANG: puncak pohon di y = 2,99 m sementara mata di
    // y = 0,6 m. Dengan fovY 45 derajat, pada jarak 1,95 m batas atas layar cuma
    // sampai y = 1,41 m — jadi dari titik spawn kamu akan melihat batang dan
    // cabang bawah saja. Mundur ke sekitar 6 m atau mendongak untuk melihat
    // seluruh mahkotanya.
    constexpr float       MESH_X         = -0.12f;
    constexpr float       MESH_Y         =  0.0f;    // alas menempel lantai
    constexpr float       MESH_Z         = -1.95f;

    constexpr bool  TEST_CUBE_ENABLED = true;
    constexpr float TEST_CUBE_X       =  0.0f;   // pusat, meter
    constexpr float TEST_CUBE_Z       = -3.0f;
    constexpr float TEST_CUBE_SIZE    =  0.5f;   // tapak persegi, meter
    constexpr float TEST_CUBE_HEIGHT  =  4.0f;   // tinggi, meter (alas di y=0)
    constexpr float AABB_PADDING            = 0.05f; // padding untuk shadow pop-in

    // Validation & Debug
    // v72: MATI. Driver vendor Android tidak mengirim layer validasi —
    // `vulkaninfo` pada perangkat ini melaporkan daftar Layers KOSONG, dan
    // VK_EXT_debug_utils juga tidak ada (yang ada cuma VK_EXT_debug_report).
    //
    // Dengan true, program berhenti di baris pertama initVulkan():
    //     throw std::runtime_error("Validation layers requested, but not available")
    constexpr bool ENABLE_VALIDATION        = false;
    constexpr bool ENABLE_VSYNC             = true;

    const std::string APP_NAME  = "Vulkan FPS - Integrated Performance Engine";
    const std::string ENGINE_NAME = "IntegratedEngine";
}

// =============================================================================
// UTILITY MACROS
// =============================================================================
#define VK_CHECK(call) \
    do { \
        vk::Result result = call; \
        if (result != vk::Result::eSuccess) { \
            throw std::runtime_error("Vulkan error at " + std::string(__FILE__) + \
                ":" + std::to_string(__LINE__) + " - Error code: " + \
                std::to_string(static_cast<int>(result))); \
        } \
    } while(0)

#define SAFE_DELETE(ptr) do { if(ptr) { delete ptr; ptr = nullptr; } } while(0)
#define ALIGN_UP(value, alignment) (((value) + (alignment) - 1) & ~((alignment) - 1))

// =============================================================================
// LOGGING SYSTEM
// =============================================================================
enum class LogLevel { Trace, Debug, Info, Warning, Error, Fatal };

class Logger {
public:
    static Logger& Get() {
        static Logger instance;
        return instance;
    }

    void Log(LogLevel level, const std::string& category, const std::string& message) {
        const char* levelStrings[] = { "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL" };
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char timeStr[26];
#ifdef _WIN32
        ctime_s(timeStr, sizeof(timeStr), &time);
#else
        ctime_r(&time, timeStr);
#endif
        timeStr[24] = '\0';

        std::lock_guard<std::mutex> lk(m_mutex);
        // v74: lewat SDL_Log, bukan std::cout.
        //
        // Di Android std::cout TIDAK ke mana-mana — ia dibuang. SDL_Log menulis
        // ke logcat, dan di Linux/Termux ia tetap ke stderr, jadi keluarannya
        // sama seperti sebelumnya di sana. Tanpa ini, satu-satunya cara mencari
        // tahu kenapa APK gagal start adalah menebak.
        SDL_Log("[%s] [%s] [%s] %s",
                timeStr.c_str(),
                levelStrings[static_cast<int>(level)],
                category.c_str(),
                message.c_str());

        if (level == LogLevel::Fatal)
            throw std::runtime_error("Fatal error: " + message);
    }
private:
    std::mutex m_mutex;
};

#define LOG_TRACE(cat, msg) Logger::Get().Log(LogLevel::Trace,   cat, msg)
#define LOG_DEBUG(cat, msg) Logger::Get().Log(LogLevel::Debug,   cat, msg)
#define LOG_INFO(cat,  msg) Logger::Get().Log(LogLevel::Info,    cat, msg)
#define LOG_WARN(cat,  msg) Logger::Get().Log(LogLevel::Warning, cat, msg)
#define LOG_ERROR(cat, msg) Logger::Get().Log(LogLevel::Error,   cat, msg)
#define LOG_FATAL(cat, msg) Logger::Get().Log(LogLevel::Fatal,   cat, msg)

// =============================================================================
// THREAD POOL — Production Priority Queue (extended dari main.cpp)
// Fitur: submit, submitPriority, submitWithArgs, submitBatchAndWait, waitAll
// =============================================================================
enum class TaskPriority : uint8_t {
    Critical = 0,   // Harus selesai sebelum GPU submit
    High     = 1,   // Frustum culling, CSM matrices, UBO update
    Normal   = 2,   // Game logic, animation
    Low      = 3    // Streaming, stats, cleanup
};

// Alias untuk kompatibilitas dengan JobSystem
using JobPriority = TaskPriority;
static constexpr JobPriority JobPriority_Critical = TaskPriority::Critical;
static constexpr JobPriority JobPriority_High     = TaskPriority::High;
static constexpr JobPriority JobPriority_Normal   = TaskPriority::Normal;
static constexpr JobPriority JobPriority_Low      = TaskPriority::Low;

struct PrioritizedTask {
    std::function<void()> func;
    TaskPriority          priority;
    uint64_t              sequenceNumber; // Tiebreaker FIFO per priority

    bool operator>(const PrioritizedTask& o) const {
        if (priority != o.priority)
            return static_cast<uint8_t>(priority) > static_cast<uint8_t>(o.priority);
        return sequenceNumber > o.sequenceNumber;
    }
};

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 0) {
        size_t count = (numThreads == 0)
            ? std::max(1u, std::thread::hardware_concurrency() - 1)
            : numThreads;
        m_workers.reserve(count);
        for (size_t i = 0; i < count; ++i)
            m_workers.emplace_back([this] { workerLoop(); });
        LOG_INFO("ThreadPool",
            "Created priority pool with " + std::to_string(count) + " threads");
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_stopping = true;
        }
        m_cv.notify_all();
        for (auto& t : m_workers)
            if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename F>
    auto submit(F&& f, TaskPriority priority = TaskPriority::Normal)
        -> std::future<std::invoke_result_t<F>>
    {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            if (m_stopping)
                throw std::runtime_error("ThreadPool: submit on stopped pool");
            m_tasks.push(PrioritizedTask{
                [task]() { (*task)(); },
                priority,
                m_sequenceCounter++
            });
        }
        m_cv.notify_one();
        return result;
    }

    // Explicit priority overload — dipakai JobSystem
    template<typename F>
    auto submitPriority(F&& f, TaskPriority priority)
        -> std::future<std::invoke_result_t<F>>
    {
        return submit(std::forward<F>(f), priority);
    }

    // Bind args ke nullary callable sebelum submit
    template<typename F, typename... Args>
    auto submitWithArgs(TaskPriority priority, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        auto boundFunc = [f = std::forward<F>(f),
                          args = std::make_tuple(std::forward<Args>(args)...)]() mutable
        {
            return std::apply(std::move(f), std::move(args));
        };
        return submit(std::move(boundFunc), priority);
    }

    // Submit sekumpulan task dan tunggu semuanya selesai di tempat
    void submitBatchAndWait(std::vector<std::function<void()>>& tasks,
                            TaskPriority priority = TaskPriority::Normal) {
        if (tasks.empty()) return;

        struct WaitState {
            std::mutex              mtx;
            std::condition_variable cv;
            std::atomic<size_t>     remaining;
            bool                    done = false;
            explicit WaitState(size_t n) : remaining(n) {}
        };
        auto state = std::make_shared<WaitState>(tasks.size());

        for (auto& task : tasks) {
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                if (m_stopping) break;
                uint64_t seq = m_sequenceCounter++;
                m_tasks.push(PrioritizedTask{
                    [taskCopy = task, state]() mutable {
                        taskCopy();
                        if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            std::unique_lock<std::mutex> lk(state->mtx);
                            state->done = true;
                            state->cv.notify_all();
                        }
                    },
                    priority,
                    seq
                });
            }
            m_cv.notify_one();
        }

        std::unique_lock<std::mutex> lock(state->mtx);
        state->cv.wait(lock, [&state] { return state->done; });
    }

    void waitAll() {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_doneCV.wait(lock, [this] {
            return m_tasks.empty() && m_activeTasks == 0;
        });
    }

    size_t threadCount()      const { return m_workers.size(); }
    size_t pendingTaskCount() const {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        return m_tasks.size();
    }

private:
    void workerLoop() {
        while (true) {
            PrioritizedTask task;
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_cv.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
                if (m_stopping && m_tasks.empty()) return;
                task = std::move(const_cast<PrioritizedTask&>(m_tasks.top()));
                m_tasks.pop();
                ++m_activeTasks;
            }
            task.func();
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                --m_activeTasks;
            }
            m_doneCV.notify_all();
        }
    }

    std::vector<std::thread> m_workers;
    std::priority_queue<
        PrioritizedTask,
        std::vector<PrioritizedTask>,
        std::greater<PrioritizedTask>>    m_tasks;
    mutable std::mutex                    m_queueMutex;
    std::condition_variable               m_cv;
    std::condition_variable               m_doneCV;
    bool                                  m_stopping       = false;
    uint64_t                              m_sequenceCounter = 0;
    std::atomic<int>                      m_activeTasks{0};
};

// =============================================================================
// JOB HANDLE + JOB SYSTEM — FC6 Style (dari main.cpp)
// Dependency graph, batch submit, per-frame cleanup
// =============================================================================
struct JobHandle {
    uint64_t                 id     = 0;
    std::shared_future<void> future;

    bool isValid() const { return id != 0; }

    void wait() const {
        if (future.valid()) future.wait();
    }

    bool isDone() const {
        if (!future.valid()) return true;
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
};

static const JobHandle kNullJob{};

class JobSystem {
public:
    explicit JobSystem(ThreadPool& pool) : m_pool(pool) {
        LOG_INFO("JobSystem",
            "Initialized with " + std::to_string(pool.threadCount()) +
            " workers, priority queue enabled");
    }

    JobSystem(const JobSystem&)            = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    template<typename F>
    JobHandle submit(F&&                    func,
                     JobPriority            priority = TaskPriority::Normal,
                     std::vector<JobHandle> deps     = {})
    {
        uint64_t id = m_nextId.fetch_add(1, std::memory_order_relaxed) + 1;

        auto taskFunc = std::make_shared<std::function<void()>>(std::forward<F>(func));
        auto depFutures = std::make_shared<std::vector<std::shared_future<void>>>();
        depFutures->reserve(deps.size());
        for (auto& d : deps)
            if (d.isValid() && d.future.valid())
                depFutures->push_back(d.future);

        auto pt = std::make_shared<std::packaged_task<void()>>(
            [taskFunc, depFutures]() {
                for (auto& f : *depFutures)
                    if (f.valid()) f.wait();
                (*taskFunc)();
            });

        std::shared_future<void> sf = pt->get_future().share();
        {
            std::lock_guard<std::mutex> lock(m_registryMutex);
            m_activeJobs[id] = sf;
        }

        try {
            m_pool.submitPriority([pt]() { (*pt)(); }, priority);
        } catch (std::exception& e) {
            LOG_ERROR("JobSystem", "Failed to submit job: " + std::string(e.what()));
        }

        JobHandle handle;
        handle.id     = id;
        handle.future = sf;
        return handle;
    }

    template<typename F>
    JobHandle submitBatch(std::vector<F>         funcs,
                          JobPriority            priority = TaskPriority::Normal,
                          std::vector<JobHandle> deps     = {})
    {
        if (funcs.empty()) return kNullJob;
        std::vector<JobHandle> batchHandles;
        batchHandles.reserve(funcs.size());
        for (auto& f : funcs)
            batchHandles.push_back(submit(std::move(f), priority, deps));
        return submit([batchHandles]() {
            for (auto& h : batchHandles) h.wait();
        }, priority);
    }

    void wait(const JobHandle& handle) { if (handle.isValid()) handle.wait(); }

    void waitAll() {
        std::vector<std::shared_future<void>> toWait;
        {
            std::lock_guard<std::mutex> lock(m_registryMutex);
            toWait.reserve(m_activeJobs.size());
            for (auto& [id, f] : m_activeJobs)
                if (f.valid()) toWait.push_back(f);
        }
        for (auto& f : toWait) if (f.valid()) f.wait();
    }

    // Hapus job yang sudah selesai di akhir frame
    void kickFrame() {
        std::lock_guard<std::mutex> lock(m_registryMutex);
        for (auto it = m_activeJobs.begin(); it != m_activeJobs.end(); ) {
            auto& f = it->second;
            bool done = !f.valid() ||
                        f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            if (done) it = m_activeJobs.erase(it);
            else      ++it;
        }
    }

    size_t activeJobCount() const {
        std::lock_guard<std::mutex> lock(m_registryMutex);
        return m_activeJobs.size();
    }

private:
    ThreadPool&                                             m_pool;
    std::atomic<uint64_t>                                  m_nextId{0};
    mutable std::mutex                                     m_registryMutex;
    std::unordered_map<uint64_t, std::shared_future<void>> m_activeJobs;
};

// =============================================================================
// SECONDARY COMMAND BUFFER + PER-FRAME THREAD DATA (dari main.cpp)
// Parallel command buffer recording per thread per frame
// =============================================================================
struct SecondaryCommandBuffer {
    vk::CommandPool   pool;
    vk::CommandBuffer buffer;
    bool              isRecording = false;
};

struct PerFrameThreadData {
    std::vector<SecondaryCommandBuffer> shadowCmdBuffers;  // shadow pass
    std::vector<SecondaryCommandBuffer> gBufferCmdBuffers; // gbuffer pass
    bool initialized = false;
};

// =============================================================================
// TIMELINE SEMAPHORE (dari main.cpp) — Vulkan 1.2 timeline
// Menggantikan binary semaphore per-pass dengan nilai timeline spesifik
// =============================================================================
struct TimelineSemaphore {
    vk::Semaphore semaphore;
    uint64_t      currentValue = 0;

    uint64_t nextSignalValue() { return ++currentValue; }

    void cpuWait(vk::Device device, uint64_t waitValue,
                 uint64_t timeout = UINT64_MAX) const {
        vk::SemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = vk::StructureType::eSemaphoreWaitInfo;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &semaphore;
        waitInfo.pValues        = &waitValue;
        (void)device.waitSemaphores(waitInfo, timeout);
    }

    uint64_t gpuQuery(vk::Device device) const {
        if (!semaphore) return 0;
        return device.getSemaphoreCounterValue(semaphore);
    }

    bool isNull() const { return !semaphore; }
};

// Helper submit dengan timeline semaphore (pNext chain pattern)
struct TimelineSubmitInfo {
    vk::SubmitInfo                  submitInfo{};
    vk::TimelineSemaphoreSubmitInfo timelineInfo{};

    std::vector<vk::Semaphore>          waitSemaphores;
    std::vector<vk::PipelineStageFlags> waitStages;
    std::vector<uint64_t>               waitValues;

    std::vector<vk::Semaphore> signalSemaphores;
    std::vector<uint64_t>      signalValues;

    std::vector<vk::CommandBuffer> commandBuffers;

    void addWaitBinary(vk::Semaphore sem, vk::PipelineStageFlags stage) {
        waitSemaphores.push_back(sem);
        waitStages.push_back(stage);
        waitValues.push_back(0);
    }

    void addWaitTimeline(vk::Semaphore sem, vk::PipelineStageFlags stage, uint64_t value) {
        waitSemaphores.push_back(sem);
        waitStages.push_back(stage);
        waitValues.push_back(value);
    }

    void addSignalBinary(vk::Semaphore sem) {
        signalSemaphores.push_back(sem);
        signalValues.push_back(0);
    }

    void addSignalTimeline(vk::Semaphore sem, uint64_t value) {
        signalSemaphores.push_back(sem);
        signalValues.push_back(value);
    }

    const vk::SubmitInfo* build() {
        timelineInfo.sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
        timelineInfo.pNext = nullptr;
        timelineInfo.waitSemaphoreValueCount   = static_cast<uint32_t>(waitValues.size());
        timelineInfo.pWaitSemaphoreValues      = waitValues.empty() ? nullptr : waitValues.data();
        timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
        timelineInfo.pSignalSemaphoreValues    = signalValues.empty() ? nullptr : signalValues.data();

        submitInfo.sType = vk::StructureType::eSubmitInfo;
        bool hasSemaphores = !waitSemaphores.empty() || !signalSemaphores.empty();
        submitInfo.pNext = hasSemaphores ? &timelineInfo : nullptr;

        submitInfo.waitSemaphoreCount  = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores     = waitSemaphores.empty() ? nullptr : waitSemaphores.data();
        submitInfo.pWaitDstStageMask   = waitStages.empty() ? nullptr : waitStages.data();
        submitInfo.commandBufferCount  = static_cast<uint32_t>(commandBuffers.size());
        submitInfo.pCommandBuffers     = commandBuffers.empty() ? nullptr : commandBuffers.data();
        submitInfo.signalSemaphoreCount= static_cast<uint32_t>(signalSemaphores.size());
        submitInfo.pSignalSemaphores   = signalSemaphores.empty() ? nullptr : signalSemaphores.data();
        return &submitInfo;
    }
};

// =============================================================================
// VULKAN STRUCTURE WRAPPERS (dari main.cpp namespace VkStructs)
// =============================================================================
namespace VkStructs {

template<typename T>
void SetPNextChain(T& base, std::vector<vk::BaseOutStructure*>& chain) {
    vk::BaseOutStructure* last = reinterpret_cast<vk::BaseOutStructure*>(&base);
    for (auto* next : chain) {
        last->pNext = next;
        last = next;
    }
    last->pNext = nullptr;
}

struct AllocatedBuffer {
    vk::Buffer              buffer = VK_NULL_HANDLE;
    vk::DeviceMemory        memory = VK_NULL_HANDLE;
    void*                   mappedData = nullptr;
    vk::DeviceSize          size = 0;
    vk::DeviceSize          alignment = 0;
    vk::MemoryPropertyFlags memoryPropertyFlags{};
};

struct AllocatedImage {
    vk::Image         image       = VK_NULL_HANDLE;
    vk::DeviceMemory  memory      = VK_NULL_HANDLE;
    vk::ImageView     imageView   = VK_NULL_HANDLE;
    vk::Format        format      = vk::Format::eUndefined;
    vk::Extent3D      extent      = {};
    uint32_t          mipLevels   = 1;
    uint32_t          arrayLayers = 1;
};

struct DescriptorSetLayoutData {
    uint32_t setNumber;
    vk::DescriptorSetLayoutCreateInfo createInfo;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
};

} // namespace VkStructs

// =============================================================================
// HALTON SEQUENCE FOR TAA JITTER (dari main.cpp)
// =============================================================================
struct HaltonSequence {
    static constexpr int SAMPLE_COUNT = 16;
    static glm::vec2 values[SAMPLE_COUNT];

    static void initialize() {
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            values[i].x = halton(i + 1, 2) - 0.5f;
            values[i].y = halton(i + 1, 3) - 0.5f;
        }
    }

private:
    static float halton(int index, int base) {
        float f = 1.0f, r = 0.0f;
        while (index > 0) {
            f /= base;
            r += f * (index % base);
            index /= base;
        }
        return r;
    }
};
glm::vec2 HaltonSequence::values[HaltonSequence::SAMPLE_COUNT];

// =============================================================================
// GPU-DRIVEN INDIRECT RENDERING (dari main.cpp)
// =============================================================================
using VkDrawIndexedIndirectCommand = ::VkDrawIndexedIndirectCommand;

struct GPUObjectDescriptor {
    glm::mat4 worldMatrix;
    glm::vec4 aabbMin;     // xyz=min, w=unused
    glm::vec4 aabbMax;     // xyz=max, w=objectIndex
    uint32_t  firstIndex;
    uint32_t  indexCount;
    int32_t   vertexOffset;
    uint32_t  flags;       // bit0=castShadow, bit1=receiveShadow
};

struct GPUCullPushConstants {
    glm::mat4 viewProj;
    glm::vec4 frustumPlanes[6];
    uint32_t  objectCount;
    uint32_t  cascadeIndex;
    float     padding[2];
};

struct IndirectDrawBuffer {
    vk::Buffer        commandBuffer   = VK_NULL_HANDLE;
    vk::DeviceMemory  commandMemory   = VK_NULL_HANDLE;
    vk::Buffer        countBuffer     = VK_NULL_HANDLE;
    vk::DeviceMemory  countMemory     = VK_NULL_HANDLE;
    uint32_t          maxDrawCount    = 0;
};

// =============================================================================
// FRUSTUM CULLING — Enhanced (unified dari main.cpp + main2.cpp)
// Menggunakan plane normalisasi + AABB center/extents projection
// =============================================================================
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    glm::vec3 GetCenter()  const { return (min + max) * 0.5f; }
    glm::vec3 GetExtents() const { return (max - min) * 0.5f; }
};

struct Plane {
    glm::vec3 normal;
    float     distance;
    float SignedDistanceTo(const glm::vec3& point) const {
        return glm::dot(normal, point) + distance;
    }
};

// Backward-compat untuk kode lama
struct FrustumPlane {
    glm::vec3 normal;
    float     d;
    float distanceTo(const glm::vec3& p) const { return glm::dot(normal, p) + d; }
};

struct Frustum {
    Plane planes[6]; // 0=left,1=right,2=bottom,3=top,4=near,5=far

    void ExtractFromMatrix(const glm::mat4& vp) {
        planes[0] = { {vp[0][3]+vp[0][0], vp[1][3]+vp[1][0], vp[2][3]+vp[2][0]}, vp[3][3]+vp[3][0] };
        planes[1] = { {vp[0][3]-vp[0][0], vp[1][3]-vp[1][0], vp[2][3]-vp[2][0]}, vp[3][3]-vp[3][0] };
        planes[2] = { {vp[0][3]+vp[0][1], vp[1][3]+vp[1][1], vp[2][3]+vp[2][1]}, vp[3][3]+vp[3][1] };
        planes[3] = { {vp[0][3]-vp[0][1], vp[1][3]-vp[1][1], vp[2][3]-vp[2][1]}, vp[3][3]-vp[3][1] };
        planes[4] = { {vp[0][2],          vp[1][2],          vp[2][2]},          vp[3][2]           };
        planes[5] = { {vp[0][3]-vp[0][2], vp[1][3]-vp[1][2], vp[2][3]-vp[2][2]}, vp[3][3]-vp[3][2] };
        for (int i = 0; i < 6; i++) {
            float len = glm::length(planes[i].normal);
            if (len > 1e-6f) { planes[i].normal /= len; planes[i].distance /= len; }
        }
    }

    void buildFromVP(const glm::mat4& vp) { ExtractFromMatrix(vp); }

    bool TestAABB(const AABB& aabb) const {
        glm::vec3 center = aabb.GetCenter(), extents = aabb.GetExtents();
        for (int i = 0; i < 6; i++) {
            float r = extents.x * std::abs(planes[i].normal.x)
                    + extents.y * std::abs(planes[i].normal.y)
                    + extents.z * std::abs(planes[i].normal.z);
            if (planes[i].SignedDistanceTo(center) + r < 0.0f) return false;
        }
        return true;
    }

    bool testAABB(const AABB& box) const { return TestAABB(box); }

    bool TestSphere(const glm::vec3& center, float radius) const {
        for (int i = 0; i < 6; i++)
            if (planes[i].SignedDistanceTo(center) + radius < 0.0f) return false;
        return true;
    }
};

// DrawCall — satu unit geometri dengan AABB untuk culling
struct DrawCall {
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t vertexOffset;
    AABB     worldAABB;
};

// Forward-declare Vertex untuk FrustumCullingManager (Vertex didefinisikan setelahnya)
struct Vertex;

// =============================================================================
// FRUSTUM CULLING MANAGER (dari main.cpp — static class)
// Per-face DrawCall building, culling, dan merge optimization
// Didefinisikan di sini sebagai forward reference; method body menggunakan
// Vertex yang sudah di-forward-declare
// =============================================================================
class FrustumCullingManager {
public:
    // Bangun AABB per-face-quad dari level geometry (6 indeks per face = 1 DrawCall)
    // Template agar bisa menerima Vertex setelah definisinya lengkap
    template<typename VertexT>
    static void BuildLevelDrawCalls(
        const std::vector<VertexT>& vertices,
        const std::vector<uint32_t>& indices,
        std::vector<DrawCall>& outDrawCalls)
    {
        outDrawCalls.clear();
        const uint32_t total = static_cast<uint32_t>(indices.size());

        // SISA TIDAK BOLEH DIBUANG.
        //
        // Versi lama menghitung numFaces = indices.size() / 6 lalu berhenti di
        // situ. Untuk geometri kotak itu selalu pas, karena tiap kuad menyumbang
        // tepat 6 indeks. Untuk mesh segitiga sungguhan TIDAK: N segitiga
        // memberi 3N indeks, dan 3N/6 = N/2 — jadi setiap mesh dengan jumlah
        // segitiga GANJIL kehilangan segitiga terakhirnya.
        //
        // Gagalnya senyap dan menyesatkan: segitiga itu tetap ada di index
        // buffer sehingga pre-pass menggambarnya dan ia tetap menghasilkan AO,
        // tapi ia hilang dari daftar gambar pass utama. Satu lubang di
        // permukaan mesh, tanpa error apa pun.
        //
        // Sekarang kelompok terakhir dipendekkan seperlunya.
        if (total % 3 != 0) {
            LOG_WARN("FrustumCulling",
                "indeks bukan kelipatan 3 (" + std::to_string(total) +
                ") - geometri kemungkinan rusak");
        }
        outDrawCalls.reserve((total + 5) / 6);

        for (uint32_t baseIdx = 0; baseIdx < total; baseIdx += 6) {
            const uint32_t groupCount = (total - baseIdx < 6u) ? (total - baseIdx) : 6u;
            AABB aabb;
            aabb.min = glm::vec3(FLT_MAX);
            aabb.max = glm::vec3(-FLT_MAX);

            // 4 unique vertices per quad
            std::set<uint32_t> usedVerts;
            for (uint32_t i = 0; i < groupCount; i++) usedVerts.insert(indices[baseIdx + i]);
            for (uint32_t vi : usedVerts) {
                aabb.min = glm::min(aabb.min, vertices[vi].pos);
                aabb.max = glm::max(aabb.max, vertices[vi].pos);
            }
            // Tambah padding supaya shadow tidak pop-in
            aabb.min -= Config::AABB_PADDING;
            aabb.max += Config::AABB_PADDING;

            DrawCall dc;
            dc.firstIndex   = baseIdx;
            dc.indexCount   = groupCount;
            dc.vertexOffset = 0;
            dc.worldAABB    = aabb;
            outDrawCalls.push_back(dc);
        }
    }

    // Cull draw calls berdasarkan frustum → isi visible index list
    static uint32_t CullDrawCalls(
        const std::vector<DrawCall>& drawCalls,
        const Frustum& frustum,
        std::vector<uint32_t>& outVisibleIndices)
    {
        outVisibleIndices.clear();
        outVisibleIndices.reserve(drawCalls.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(drawCalls.size()); i++)
            if (frustum.TestAABB(drawCalls[i].worldAABB))
                outVisibleIndices.push_back(i);
        return static_cast<uint32_t>(outVisibleIndices.size());
    }

    // Merge draw calls berurutan → kurangi vkCmdDrawIndexed overhead
    static void MergeConsecutiveDrawCalls(
        const std::vector<DrawCall>& allCalls,
        const std::vector<uint32_t>& visibleIndices,
        std::vector<DrawCall>& outMerged)
    {
        outMerged.clear();
        if (visibleIndices.empty()) return;

        DrawCall current = allCalls[visibleIndices[0]];
        for (size_t i = 1; i < visibleIndices.size(); i++) {
            const DrawCall& next = allCalls[visibleIndices[i]];
            // Merge jika indeks berurutan dan vertex offset sama
            if (next.firstIndex == current.firstIndex + current.indexCount &&
                next.vertexOffset == current.vertexOffset)
            {
                current.indexCount += next.indexCount;
                current.worldAABB.min = glm::min(current.worldAABB.min, next.worldAABB.min);
                current.worldAABB.max = glm::max(current.worldAABB.max, next.worldAABB.max);
            } else {
                outMerged.push_back(current);
                current = next;
            }
        }
        outMerged.push_back(current);
    }
};

// =============================================================================
// DATA STRUCTURES
// =============================================================================
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texCoord;   // untuk kompatibilitas extended
    glm::vec3 tangent;    // normal mapping
    glm::vec3 bitangent;  // normal mapping

    // Posisi verteks ini pada frame SEBELUMNYA, di ruang dunia.
    //
    // Untuk semua geometri statis nilainya sama dengan pos, dan diisi sekali
    // saat buffer dibangun. Untuk objek dinamis ia benar-benar berbeda, dan
    // itulah gunanya: prepass menghitung motion vector dari posisi lama yang
    // SEBENARNYA, bukan dari posisi sekarang yang diproyeksikan dengan matriks
    // kamera lama. Tanpa ini, benda bergerak melaporkan motion vector milik
    // kamera saja — TAA menyangka permukaannya diam lalu memadukannya dengan
    // riwayat dari tempat yang salah, dan hasilnya bayangan ganda.
    glm::vec3 prevPos{0.0f};

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 4> attribs = {};
        attribs[0] = {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)};
        attribs[1] = {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)};
        attribs[2] = {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)};
        attribs[3] = {3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, prevPos)};
        return attribs;
    }

    bool operator==(const Vertex& o) const {
        return pos == o.pos && normal == o.normal && color == o.color;
    }
};

struct JoystickVertex {
    glm::vec2 pos;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {0, sizeof(JoystickVertex), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 1> attribs = {};
        attribs[0] = {0, 0, vk::Format::eR32G32Sfloat, offsetof(JoystickVertex, pos)};
        return attribs;
    }
};

struct JoystickPC {
    glm::vec4 offsetAndScale;
    glm::vec4 resolution;
    glm::vec4 color;
};

// Extended UBO — kompatibel dengan main.cpp cascade shadows
struct UniformBufferObject {
    alignas(16) glm::mat4 mvp;
    alignas(16) glm::mat4 lightMVP; // primary shadow (slot 0 compat)
    alignas(16) glm::vec3 lightPos;
    alignas(4)  float ambientStrength;
    alignas(4)  float diffuseStrength;
    alignas(4)  float specularStrength;
    alignas(4)  float shininess;
    alignas(16) glm::vec3 cameraPos;   // tambahan dari main.cpp
    alignas(4)  float shadowBias;      // tambahan dari main.cpp
};

struct TouchPoint {
    SDL_FingerID fingerId;
    glm::vec2    position;
    glm::vec2    startDelta;
    bool         isActive;
    bool         isJoystick;
};

// Per-tile bounding box untuk frustum culling
struct TileInfo {
    AABB     aabb;
    uint32_t firstIndex;
    uint32_t indexCount;
    bool     visible;
};

// =============================================================================
// RENDER RESOURCE + RENDER GRAPH (dari main.cpp)
// Automatic dependency-based barrier/transition resolution
// =============================================================================
enum class PassType : uint8_t { Graphics = 0, Compute = 1, Transfer = 2 };

enum class ResourceState : uint8_t {
    Undefined, RenderTarget, DepthWrite, DepthRead,
    ShaderRead, StorageWrite, StorageReadWrite,
    TransferSrc, TransferDst, Present
};

struct RenderResource {
    vk::Image            image         = VK_NULL_HANDLE;
    vk::ImageView        imageView     = VK_NULL_HANDLE;
    vk::Format           format        = vk::Format::eUndefined;
    ResourceState        currentState  = ResourceState::Undefined;
    vk::ImageLayout      currentLayout = vk::ImageLayout::eUndefined;
    vk::ImageAspectFlags aspect        = vk::ImageAspectFlagBits::eColor;
    std::string          name;

    static vk::ImageLayout StateToLayout(ResourceState state) {
        switch (state) {
            case ResourceState::RenderTarget:     return vk::ImageLayout::eColorAttachmentOptimal;
            case ResourceState::DepthWrite:       return vk::ImageLayout::eDepthStencilAttachmentOptimal;
            case ResourceState::DepthRead:        return vk::ImageLayout::eShaderReadOnlyOptimal;
            case ResourceState::ShaderRead:       return vk::ImageLayout::eShaderReadOnlyOptimal;
            case ResourceState::StorageWrite:     return vk::ImageLayout::eGeneral;
            case ResourceState::StorageReadWrite: return vk::ImageLayout::eGeneral;
            case ResourceState::TransferSrc:      return vk::ImageLayout::eTransferSrcOptimal;
            case ResourceState::TransferDst:      return vk::ImageLayout::eTransferDstOptimal;
            case ResourceState::Present:          return vk::ImageLayout::ePresentSrcKHR;
            default:                              return vk::ImageLayout::eUndefined;
        }
    }

    static vk::AccessFlags StateToAccess(ResourceState state) {
        switch (state) {
            case ResourceState::RenderTarget:
                return vk::AccessFlagBits::eColorAttachmentWrite;
            case ResourceState::DepthWrite:
                return vk::AccessFlagBits::eDepthStencilAttachmentWrite
                     | vk::AccessFlagBits::eDepthStencilAttachmentRead;
            case ResourceState::DepthRead:
            case ResourceState::ShaderRead:
                return vk::AccessFlagBits::eShaderRead;
            case ResourceState::StorageWrite:
                return vk::AccessFlagBits::eShaderWrite;
            case ResourceState::StorageReadWrite:
                return vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
            case ResourceState::TransferSrc: return vk::AccessFlagBits::eTransferRead;
            case ResourceState::TransferDst: return vk::AccessFlagBits::eTransferWrite;
            default:                         return {};
        }
    }

    static vk::PipelineStageFlags StateToPipelineStage(ResourceState state, PassType passType) {
        switch (state) {
            case ResourceState::RenderTarget:
                return vk::PipelineStageFlagBits::eColorAttachmentOutput;
            case ResourceState::DepthWrite:
                return vk::PipelineStageFlagBits::eEarlyFragmentTests
                     | vk::PipelineStageFlagBits::eLateFragmentTests;
            case ResourceState::DepthRead:
                return vk::PipelineStageFlagBits::eFragmentShader
                     | vk::PipelineStageFlagBits::eComputeShader;
            case ResourceState::ShaderRead:
                if (passType == PassType::Compute)
                    return vk::PipelineStageFlagBits::eComputeShader;
                return vk::PipelineStageFlagBits::eFragmentShader
                     | vk::PipelineStageFlagBits::eComputeShader;
            case ResourceState::StorageWrite:
            case ResourceState::StorageReadWrite:
                return vk::PipelineStageFlagBits::eComputeShader;
            case ResourceState::TransferSrc:
            case ResourceState::TransferDst:
                return vk::PipelineStageFlagBits::eTransfer;
            default:
                return vk::PipelineStageFlagBits::eTopOfPipe;
        }
    }
};

struct ResourceTransition {
    RenderResource* resource  = nullptr;
    ResourceState   stateBefore;
    ResourceState   stateAfter;
    PassType        passType  = PassType::Graphics;
};

struct RenderPass {
    std::string                           name;
    PassType                              type     = PassType::Graphics;
    std::vector<ResourceTransition>       transitions;
    std::function<void(vk::CommandBuffer)>recordFunc;
    uint64_t                              timelineSignalValue = 0;

    void addTransition(RenderResource* res, ResourceState before,
                       ResourceState after, PassType pt = PassType::Graphics) {
        transitions.push_back({res, before, after, pt});
    }
};

// RenderGraph — automatic barrier emission (FC6 style)
class RenderGraph {
public:
    static void EmitBarriers(vk::CommandBuffer cb, const RenderPass& pass) {
        std::vector<vk::ImageMemoryBarrier> barriers;
        barriers.reserve(pass.transitions.size() * 2);

        vk::PipelineStageFlags srcStages = vk::PipelineStageFlagBits::eTopOfPipe;
        vk::PipelineStageFlags dstStages = vk::PipelineStageFlagBits::eBottomOfPipe;

        for (const auto& t : pass.transitions) {
            if (!t.resource || !t.resource->image) continue;

            vk::ImageLayout desiredLayout = RenderResource::StateToLayout(t.stateAfter);
            if (t.resource->currentLayout == desiredLayout) {
                t.resource->currentState = t.stateAfter;
                continue;
            }

            vk::ImageMemoryBarrier barrier{};
            barrier.sType               = vk::StructureType::eImageMemoryBarrier;
            barrier.oldLayout           = t.resource->currentLayout;
            barrier.newLayout           = desiredLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image               = t.resource->image;
            barrier.subresourceRange    = {t.resource->aspect, 0, 1, 0, 1};
            barrier.srcAccessMask       = RenderResource::StateToAccess(t.stateBefore);
            barrier.dstAccessMask       = RenderResource::StateToAccess(t.stateAfter);

            srcStages |= RenderResource::StateToPipelineStage(t.stateBefore, t.passType);
            dstStages |= RenderResource::StateToPipelineStage(t.stateAfter, t.passType);

            if (t.resource->currentLayout == vk::ImageLayout::eUndefined) {
                barrier.srcAccessMask = vk::AccessFlags{};
                srcStages = vk::PipelineStageFlagBits::eTopOfPipe;
            }

            t.resource->currentState  = t.stateAfter;
            t.resource->currentLayout = desiredLayout;
            barriers.push_back(barrier);
        }

        if (!barriers.empty()) {
            if (dstStages == vk::PipelineStageFlagBits::eBottomOfPipe &&
                srcStages != vk::PipelineStageFlagBits::eTopOfPipe)
                dstStages = vk::PipelineStageFlagBits::eAllCommands;

            cb.pipelineBarrier(srcStages, dstStages, vk::DependencyFlags{},
                0, nullptr, 0, nullptr,
                static_cast<uint32_t>(barriers.size()), barriers.data());
        }
    }

    static void EmitOwnershipBarrier(
        vk::CommandBuffer cb, RenderResource* resource,
        uint32_t srcQueueFamily, uint32_t dstQueueFamily,
        vk::ImageLayout layout, vk::AccessFlags dstAccess,
        vk::PipelineStageFlags dstStage)
    {
        if (!resource || !resource->image) return;
        if (srcQueueFamily == dstQueueFamily) return;

        vk::ImageMemoryBarrier barrier{};
        barrier.sType               = vk::StructureType::eImageMemoryBarrier;
        barrier.oldLayout           = layout;
        barrier.newLayout           = layout;
        barrier.srcQueueFamilyIndex = srcQueueFamily;
        barrier.dstQueueFamilyIndex = dstQueueFamily;
        barrier.image               = resource->image;
        barrier.subresourceRange    = {resource->aspect, 0, 1, 0, 1};
        barrier.srcAccessMask       = {};
        barrier.dstAccessMask       = dstAccess;

        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, dstStage,
            vk::DependencyFlags{}, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    static void ResetResourceStates(std::vector<RenderResource*>& resources) {
        for (auto* r : resources) {
            if (r) {
                r->currentState  = ResourceState::Undefined;
                r->currentLayout = vk::ImageLayout::eUndefined;
            }
        }
    }

    static RenderPass MakeTransition(
        const std::string& name, RenderResource* resource,
        ResourceState from, ResourceState to, PassType type = PassType::Graphics)
    {
        RenderPass pass; pass.name = name; pass.type = type;
        pass.addTransition(resource, from, to, type);
        return pass;
    }
};

// UpdateEntry helper untuk state tracking
struct UpdateEntry {
    RenderResource* res;
    ResourceState   state;
    vk::ImageLayout layout;
};

// =============================================================================
// GEOMETRY GENERATOR (static class dari main.cpp)
// Forward declaration — defined after FirstPersonCamera (line ~1416)
extern std::vector<std::vector<int>> levelMap;

// =============================================================================
class GeometryGenerator {
public:
    struct MeshData {
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;
    };

    // =========================================================================
    //  Pemuat OBJ minimal
    //
    //  Menangani yang dipakai eksportir Blender secara default: v / vn / f,
    //  dengan bentuk indeks "a", "a/b", "a/b/c", dan "a//c". Indeks 1-based, dan
    //  indeks NEGATIF (relatif dari ujung) juga ditangani karena sebagian
    //  eksportir memakainya. Poligon >3 sisi dikipas jadi segitiga.
    //
    //  vt sengaja diabaikan: format Vertex belum punya UV. Begitu UV masuk,
    //  di sinilah tempatnya.
    //
    //  Kalau OBJ tidak membawa vn, normal dihitung sendiri sebagai rata-rata
    //  berbobot luas dari normal muka. Bobot luas didapat gratis karena
    //  cross() dari dua sisi segitiga panjangnya dua kali luasnya — jadi
    //  jangan dinormalkan sebelum diakumulasi.
    // =========================================================================
    struct ObjMesh {
        std::vector<glm::vec3> pos;
        std::vector<glm::vec3> nrm;
        std::vector<glm::vec3> col;   // kosong kalau berkas tidak membawa warna
        std::vector<uint32_t>  idx;
        bool hasColor = false;
    };

    static bool loadObj(const std::string& path, float scale,
                        const glm::vec3& offset, ObjMesh& out)
    {
        // v73: dimuat ke memori dulu lalu diurai dari istringstream, supaya
        // jalur parsing baris di bawah tidak berubah sama sekali.
        bool ok = false;
        auto objBytes = platformLoadFile(path, &ok);
        if (!ok) {
            LOG_WARN("Mesh", "OBJ tidak bisa dibuka: " + path);
            return false;
        }
        std::istringstream f(std::string(objBytes.begin(), objBytes.end()));

        std::vector<glm::vec3> P, N, PC;
        bool hasVertexColor = false;
        struct Key { uint32_t p, n; };
        std::map<uint64_t, uint32_t> remap;   // (p,n) -> indeks verteks keluaran

        auto resolve = [](long v, size_t count) -> long {
            return (v < 0) ? (long)count + v : v - 1;   // 1-based, negatif = relatif
        };

        std::string line;
        while (std::getline(f, line)) {
            if (line.size() < 2) continue;
            std::istringstream ss(line);
            std::string tag; ss >> tag;

            if (tag == "v") {
                // Bentuk panjang 'v x y z r g b' membawa WARNA VERTEKS. Itu
                // perluasan tidak resmi tapi dipakai luas (MeshLab, beberapa
                // addon Blender), dan di sini ia menggantikan seluruh sistem
                // material: satu mesh bisa punya batang cokelat dan daun hijau
                // tanpa perlu .mtl, tanpa tekstur, tanpa UV.
                glm::vec3 p, c(1.0f);
                ss >> p.x >> p.y >> p.z;
                if (ss >> c.r >> c.g >> c.b) { hasVertexColor = true; }
                else                         { c = glm::vec3(1.0f); }
                P.push_back(p);
                PC.push_back(c);
            } else if (tag == "vn") {
                glm::vec3 n; ss >> n.x >> n.y >> n.z; N.push_back(n);
            } else if (tag == "f") {
                std::vector<uint32_t> poly;
                std::string tok;
                while (ss >> tok) {
                    long vi = 0, ni = 0;
                    size_t s1 = tok.find('/');
                    if (s1 == std::string::npos) {
                        vi = std::atol(tok.c_str());
                    } else {
                        vi = std::atol(tok.substr(0, s1).c_str());
                        size_t s2 = tok.find('/', s1 + 1);
                        if (s2 != std::string::npos && s2 + 1 < tok.size())
                            ni = std::atol(tok.substr(s2 + 1).c_str());
                    }
                    long pi = resolve(vi, P.size());
                    long qi = (ni != 0) ? resolve(ni, N.size()) : -1;
                    if (pi < 0 || pi >= (long)P.size()) continue;
                    if (qi >= (long)N.size()) qi = -1;

                    uint64_t key = ((uint64_t)(uint32_t)pi << 32) | (uint32_t)(qi + 1);
                    auto it = remap.find(key);
                    if (it == remap.end()) {
                        uint32_t nid = static_cast<uint32_t>(out.pos.size());
                        out.pos.push_back(P[pi] * scale + offset);
                        out.nrm.push_back(qi >= 0 ? N[qi] : glm::vec3(0.0f));
                        out.col.push_back(PC[pi]);
                        remap.emplace(key, nid);
                        poly.push_back(nid);
                    } else {
                        poly.push_back(it->second);
                    }
                }
                for (size_t k = 2; k < poly.size(); ++k) {   // kipas
                    out.idx.push_back(poly[0]);
                    out.idx.push_back(poly[k - 1]);
                    out.idx.push_back(poly[k]);
                }
            }
        }

        if (out.pos.empty() || out.idx.size() < 3) {
            LOG_WARN("Mesh", "OBJ kosong atau tanpa segitiga: " + path);
            return false;
        }

        // Normal hilang -> hitung halus, berbobot luas.
        bool needNormals = N.empty();
        if (!needNormals) {
            for (const auto& n : out.nrm)
                if (glm::dot(n, n) < 1e-12f) { needNormals = true; break; }
        }
        if (needNormals) {
            std::vector<glm::vec3> acc(out.pos.size(), glm::vec3(0.0f));
            for (size_t t = 0; t + 2 < out.idx.size(); t += 3) {
                uint32_t a = out.idx[t], b = out.idx[t+1], c = out.idx[t+2];
                glm::vec3 fn = glm::cross(out.pos[b] - out.pos[a],
                                          out.pos[c] - out.pos[a]);  // |fn| = 2 x luas
                acc[a] += fn; acc[b] += fn; acc[c] += fn;
            }
            for (size_t i = 0; i < out.nrm.size(); ++i) {
                out.nrm[i] = (glm::dot(acc[i], acc[i]) > 1e-12f)
                           ? glm::normalize(acc[i]) : glm::vec3(0, 1, 0);
            }
        } else {
            for (auto& n : out.nrm) n = glm::normalize(n);
        }

        out.hasColor = hasVertexColor;

        LOG_INFO("Mesh", "OBJ dimuat: " + path + " -> " +
                 std::to_string(out.pos.size()) + " verteks, " +
                 std::to_string(out.idx.size() / 3) + " segitiga" +
                 (needNormals ? " (normal dihitung sendiri)" : "") +
                 (hasVertexColor ? ", warna verteks dari berkas" : ""));
        return true;
    }

    static void generateLevelGeometry(std::vector<Vertex>& outVertices,
                                       std::vector<uint32_t>& outIndices,
                                       std::vector<TileInfo>* outTiles = nullptr)
    {
        struct FaceDef {
            glm::vec3 normal;
            std::array<glm::vec3, 4> corners;
        };

        const std::array<FaceDef, 6> cubeFaces = {
            FaceDef{ {0,0,1},  {{ {-0.5f, 0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, {-0.5f,-0.5f, 0.5f} }} },
            FaceDef{ {0,0,-1}, {{ { 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f} }} },
            FaceDef{ {0,1,0},  {{ {-0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f} }} },
            FaceDef{ {0,-1,0}, {{ {-0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f} }} },
            FaceDef{ {1,0,0},  {{ { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f,-0.5f} }} },
            FaceDef{ {-1,0,0}, {{ {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f} }} }
        };

        // ---- SATU-SATUNYA jalan menambahkan geometri ----------------------
        //
        // emitBox menulis verteks, indeks, DAN TileInfo dalam satu panggilan.
        // Itu bukan sekadar kerapian: pass utama membangun daftar gambarnya
        // lewat BuildLevelDrawCalls() yang menyapu index buffer, jadi objek
        // baru otomatis terlihat di layar — tapi pass cascade beriterasi atas
        // m_tiles. Kalau pendaftaran tile ditulis terpisah dan terlupa, objeknya
        // terender sempurna namun TIDAK melempar bayangan, sementara pre-pass
        // tetap menggambarnya sehingga ia tetap menghasilkan AO. Gejala seperti
        // itu menyesatkan dan mahal dikejar. Dengan emitBox, melupakannya jadi
        // mustahil karena tidak ada jalur lain.
        //
        // faceMask memilih muka: bit 0..5 sesuai urutan cubeFaces
        // (0 depan +Z, 1 belakang -Z, 2 atas +Y, 3 alas -Y, 4 kanan +X, 5 kiri -X).
        auto emitBox = [&](const glm::vec3& center,
                           const glm::vec3& halfExtents,
                           const glm::vec3& sideCol,
                           const glm::vec3& topCol,
                           uint32_t faceMask)
        {
            TileInfo tile{};
            if (outTiles) {
                tile.aabb.min   = center - halfExtents;
                tile.aabb.max   = center + halfExtents;
                tile.firstIndex = static_cast<uint32_t>(outIndices.size());
            }

            for (int fi = 0; fi < 6; ++fi) {
                if ((faceMask & (1u << fi)) == 0) continue;
                const auto& face = cubeFaces[fi];
                uint32_t vb = static_cast<uint32_t>(outVertices.size());
                for (int k = 0; k < 4; k++) {
                    Vertex v{};
                    v.pos    = center + face.corners[k] * (halfExtents * 2.0f);
                    v.color  = (fi == 2) ? topCol : sideCol;
                    v.normal = face.normal;
                    outVertices.push_back(v);
                }
                outIndices.push_back((uint32_t)(vb+0)); outIndices.push_back((uint32_t)(vb+1));
                outIndices.push_back((uint32_t)(vb+2)); outIndices.push_back((uint32_t)(vb+2));
                outIndices.push_back((uint32_t)(vb+3)); outIndices.push_back((uint32_t)(vb+0));
            }

            if (outTiles) {
                tile.indexCount = static_cast<uint32_t>(outIndices.size()) - tile.firstIndex;
                tile.visible    = true;
                outTiles->push_back(tile);
            }
        };

        // ---- Mesh segitiga sembarang, jalur yang sama dengan emitBox --------
        //
        // Menerima posisi + normal + indeks yang SUDAH dalam ruang dunia, lalu
        // menulis verteks, indeks, dan TileInfo-nya. Sama seperti emitBox, ini
        // satu-satunya pintu — pendaftaran tile tidak bisa terlupa.
        //
        // Indeks digeser dengan basis verteks saat ini karena seluruh level
        // berbagi satu vertex buffer.
        auto emitMesh = [&](const std::vector<glm::vec3>& pos,
                            const std::vector<glm::vec3>& nrm,
                            const std::vector<glm::vec3>& col,   // boleh kosong
                            const std::vector<uint32_t>&  idx,
                            const glm::vec3& fallbackColor)
        {
            if (pos.empty() || idx.size() < 3) return;

            TileInfo tile{};
            const uint32_t vb = static_cast<uint32_t>(outVertices.size());
            if (outTiles) {
                tile.aabb.min   = glm::vec3( FLT_MAX);
                tile.aabb.max   = glm::vec3(-FLT_MAX);
                tile.firstIndex = static_cast<uint32_t>(outIndices.size());
            }

            for (size_t i = 0; i < pos.size(); ++i) {
                Vertex v{};
                v.pos    = pos[i];
                v.color  = (i < col.size()) ? col[i] : fallbackColor;
                v.normal = (i < nrm.size()) ? nrm[i] : glm::vec3(0, 1, 0);
                outVertices.push_back(v);
                if (outTiles) {
                    tile.aabb.min = glm::min(tile.aabb.min, v.pos);
                    tile.aabb.max = glm::max(tile.aabb.max, v.pos);
                }
            }
            for (uint32_t i : idx) outIndices.push_back(vb + i);

            if (outTiles) {
                tile.indexCount = static_cast<uint32_t>(outIndices.size()) - tile.firstIndex;
                tile.visible    = true;
                outTiles->push_back(tile);
            }
        };

        const float ms = Config::MAP_SCALE;
        const float wh = Config::WALL_HEIGHT;
        const int   mH = (int)levelMap.size();
        const int   mW = (int)levelMap[0].size();
        const float ox = -(mW * ms) / 2.0f;
        const float oz = -(mH * ms) / 2.0f;

        const glm::vec3 sideColor(0.7f, 0.7f, 0.7f);
        const glm::vec3 topColor (0.85f, 0.85f, 0.85f);

        for (int z = 0; z < mH; z++) {
            for (int x = 0; x < mW; x++) {
                const int lvl = levelMap[z][x];
                if (lvl <= 0) continue;

                // v41: nilai sel adalah KODE TINGGI, bukan bendera 0/1.
                const float h = wh * static_cast<float>(lvl);

                glm::vec3 worldPos;
                worldPos.x = ox + (x * ms) + (ms / 2.0f);
                worldPos.y = h / 2.0f;
                worldPos.z = oz + (z * ms) + (ms / 2.0f);

                // Muka samping hanya boleh disembunyikan kalau tetangganya
                // SAMA TINGGI. Dengan bangunan bertingkat, tetangga yang lebih
                // pendek meninggalkan bidang terbuka di atasnya — kalau tetap
                // disembunyikan, dindingnya bolong dan bayangannya ikut bocor.
                auto sameH = [&](int zz, int xx) {
                    if (zz < 0 || zz >= mH || xx < 0 || xx >= mW) return false;
                    return levelMap[zz][xx] == lvl;
                };

                // Muka alas (bit 3) tidak pernah dipakai: ia terkubur di lantai.
                uint32_t mask = (1u << 2);                       // atas, selalu
                if (!sameH(z+1, x)) mask |= (1u << 0);
                if (!sameH(z-1, x)) mask |= (1u << 1);
                if (!sameH(z, x+1)) mask |= (1u << 4);
                if (!sameH(z, x-1)) mask |= (1u << 5);

                emitBox(worldPos, glm::vec3(ms/2, h/2, ms/2), sideColor, topColor, mask);
            }
        }

        // ---- Kubus uji yang berdiri sendiri --------------------------------
        // Objek ini TIDAK berasal dari levelMap. Ia sengaja ditulis lewat jalur
        // yang sama persis dengan kubus dinding — vertex buffer yang sama, index
        // buffer yang sama, dan yang paling penting: TileInfo-nya didaftarkan.
        //
        // Pendaftaran tile itu bagian yang mudah terlupa, dan kalau terlupa
        // gejalanya menyesatkan. Pass utama membangun daftar gambarnya lewat
        // BuildLevelDrawCalls(), yang MENYAPU index buffer per 6 indeks, jadi
        // objek baru otomatis ikut terlihat di layar. Tapi pass cascade
        // beriterasi atas m_tiles. Tanpa baris di bawah, kubus ini akan terender
        // sempurna namun TIDAK melempar bayangan sama sekali — dan karena
        // pre-pass menggambar seluruh index buffer tanpa culling, ia tetap
        // menghasilkan AO dan contact shadow. Itu kombinasi yang membingungkan
        // kalau tidak tahu sebabnya.
        if (Config::TEST_CUBE_ENABLED) {
            const float cs = Config::TEST_CUBE_SIZE;     // tapak (x dan z)
            const float ch = Config::TEST_CUBE_HEIGHT;   // tinggi (y)

            // Muka alas (bit 3) dilewati: ia sebidang dengan lantai, jadi kalau
            // ikut dirasterisasi ke shadow map keduanya berebut kedalaman di
            // texel yang sama. Lima muka -> 20 verteks, 30 indeks — kelipatan 6,
            // syarat BuildLevelDrawCalls yang mengelompokkan indeks per 6.
            const uint32_t mask = 0x3Fu & ~(1u << 3);

            emitBox(glm::vec3(Config::TEST_CUBE_X,
                              ch * 0.5f,                 // alas menempel di y=0
                              Config::TEST_CUBE_Z),
                    glm::vec3(cs * 0.5f, ch * 0.5f, cs * 0.5f),
                    glm::vec3(0.55f),                    // lebih gelap dari dinding 0,70
                    glm::vec3(0.65f),
                    mask);
        }

        // ---- Mesh OBJ opsional ----------------------------------------------
        // Cabang "dimatikan" SENGAJA ikut menulis log. Tanpa itu, path kosong
        // membuat seluruh fitur diam total — tidak ada mesh, tidak ada pesan,
        // tidak ada petunjuk kenapa. Keadaan mati harus terlihat sama jelasnya
        // dengan keadaan gagal.
        if (Config::MESH_OBJ_PATH[0] == '\0') {
            LOG_INFO("Mesh", "MESH_OBJ_PATH kosong - pemuatan mesh dilewati");
        } else {
            ObjMesh m;
            if (loadObj(Config::MESH_OBJ_PATH, Config::MESH_SCALE,
                        glm::vec3(Config::MESH_X, Config::MESH_Y, Config::MESH_Z), m)) {
                const uint32_t triBefore = static_cast<uint32_t>(outIndices.size()) / 3;
                emitMesh(m.pos, m.nrm,
                         m.hasColor ? m.col : std::vector<glm::vec3>{},
                         m.idx,
                         glm::vec3(0.60f, 0.62f, 0.58f));
                LOG_INFO("Mesh", "mesh masuk scene: +" +
                         std::to_string(static_cast<uint32_t>(outIndices.size()) / 3 - triBefore) +
                         " segitiga, total " +
                         std::to_string(static_cast<uint32_t>(outIndices.size()) / 3));
            }
        }
    }
};

// =============================================================================
// SHADER MODULE — static utility class (dari main.cpp)
// =============================================================================
class ShaderModule {
public:
    static vk::ShaderModule Create(vk::Device device, const std::vector<char>& code) {
        if (code.empty()) return VK_NULL_HANDLE;
        vk::ShaderModuleCreateInfo ci{};
        ci.sType    = vk::StructureType::eShaderModuleCreateInfo;
        ci.codeSize = code.size();
        ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        return device.createShaderModule(ci);
    }

    static std::vector<char> ReadFile(const std::string& filename) {
        bool ok = false;
        auto buffer = platformLoadFile(filename, &ok);   // v73: aset APK juga
        if (!ok) {
            LOG_WARN("Shader", "Failed to open shader file: " + filename);
            return {};
        }
        return buffer;
    }
};

// =============================================================================
// VULKAN MEMORY ALLOCATOR — RAII allocation helper (dari main.cpp)
// =============================================================================
class VulkanMemoryAllocator {
public:
    VulkanMemoryAllocator(vk::PhysicalDevice physicalDevice, vk::Device device)
        : m_physicalDevice(physicalDevice), m_device(device) {
        m_memProps = physicalDevice.getMemoryProperties();
    }

    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < m_memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) &&
                (m_memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        throw std::runtime_error("No suitable memory type");
    }

    VkStructs::AllocatedBuffer CreateBuffer(vk::DeviceSize size,
                                             vk::BufferUsageFlags usage,
                                             vk::MemoryPropertyFlags properties) {
        VkStructs::AllocatedBuffer buf{};
        buf.size = size; buf.memoryPropertyFlags = properties;

        vk::BufferCreateInfo bi{};
        bi.size = size; bi.usage = usage; bi.sharingMode = vk::SharingMode::eExclusive;
        buf.buffer = m_device.createBuffer(bi);

        auto mr = m_device.getBufferMemoryRequirements(buf.buffer);
        buf.alignment = mr.alignment;

        vk::MemoryAllocateInfo ai{};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = FindMemoryType(mr.memoryTypeBits, properties);
        buf.memory = m_device.allocateMemory(ai);
        m_device.bindBufferMemory(buf.buffer, buf.memory, 0);

        if (properties & vk::MemoryPropertyFlagBits::eHostVisible)
            buf.mappedData = m_device.mapMemory(buf.memory, 0, size);

        return buf;
    }

    void CopyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
                    vk::CommandPool commandPool, vk::Queue queue) {
        vk::CommandBufferAllocateInfo ai{};
        ai.level = vk::CommandBufferLevel::ePrimary;
        ai.commandPool = commandPool; ai.commandBufferCount = 1;
        auto cb = m_device.allocateCommandBuffers(ai)[0];

        vk::CommandBufferBeginInfo bi{};
        bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        cb.begin(bi);
        vk::BufferCopy cr{}; cr.size = size;
        cb.copyBuffer(src, dst, 1, &cr);
        cb.end();

        vk::SubmitInfo si{}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        queue.submit(si, VK_NULL_HANDLE);
        queue.waitIdle();
        m_device.freeCommandBuffers(commandPool, cb);
    }

    void DestroyBuffer(const VkStructs::AllocatedBuffer& buffer) {
        if (buffer.mappedData) m_device.unmapMemory(buffer.memory);
        if (buffer.buffer)     m_device.destroyBuffer(buffer.buffer);
        if (buffer.memory)     m_device.freeMemory(buffer.memory);
    }

private:
    vk::PhysicalDevice                   m_physicalDevice;
    vk::Device                           m_device;
    vk::PhysicalDeviceMemoryProperties   m_memProps;
};

// =============================================================================
// FIRST PERSON CAMERA — Quaternion-based (dari main.cpp)
// Menggantikan kamera inline di HelloTriangleApplication dengan class terpisah
// yang reusable dan clean
// =============================================================================
class FirstPersonCamera {
public:
    FirstPersonCamera() { Reset(); }

    void Reset() {
        m_position    = glm::vec3(0.0f, Config::PLAYER_HEIGHT, 0.0f);
        m_orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        m_yaw         = 0.0f;
        m_pitch       = 0.0f;
        m_fov         = glm::radians(Config::CAMERA_FOV);
        m_nearPlane   = Config::CAMERA_NEAR_PLANE;
        m_farPlane    = Config::CAMERA_FAR_PLANE;
        // v73: dari ukuran SEBENARNYA, bukan konstanta. Di layar HP 1080x2340
        // rasionya 0,46 — memakai 800/600 = 1,33 akan membuat gambar gepeng.
        m_aspectRatio = static_cast<float>(g_windowWidth)
                      / static_cast<float>(g_windowHeight);
    }

    void SetPosition(const glm::vec3& pos)  { m_position    = pos; }
    void SetAspectRatio(float aspectRatio)  { m_aspectRatio = aspectRatio; }

    void Rotate(float deltaYaw, float deltaPitch) {
        m_yaw   -= deltaYaw;
        m_pitch -= deltaPitch;
        const float maxPitch = 89.0f * (float)M_PI / 180.0f;
        m_pitch = std::clamp(m_pitch, -maxPitch, maxPitch);
        glm::quat qPitch = glm::angleAxis(m_pitch, glm::vec3(1, 0, 0));
        glm::quat qYaw   = glm::angleAxis(m_yaw,   glm::vec3(0, 1, 0));
        m_orientation    = glm::normalize(qYaw * qPitch);
    }

    // Setters untuk sync dengan input joystick
    void SetYaw(float yaw)     { m_yaw   = yaw; }
    void SetPitch(float pitch) { m_pitch = pitch; }
    void UpdateOrientation() {
        const float maxPitch = 89.0f * (float)M_PI / 180.0f;
        m_pitch = std::clamp(m_pitch, -maxPitch, maxPitch);
        glm::quat qPitch = glm::angleAxis(m_pitch, glm::vec3(1, 0, 0));
        glm::quat qYaw   = glm::angleAxis(m_yaw,   glm::vec3(0, 1, 0));
        m_orientation    = glm::normalize(qYaw * qPitch);
    }

    glm::vec3 GetPosition()    const { return m_position; }
    glm::quat GetOrientation() const { return m_orientation; }
    float     GetYaw()         const { return m_yaw; }
    float     GetPitch()       const { return m_pitch; }

    glm::vec3 GetForward() const { return m_orientation * glm::vec3(0, 0, -1); }
    glm::vec3 GetRight()   const { return m_orientation * glm::vec3(1, 0,  0); }
    glm::vec3 GetUp()      const { return m_orientation * glm::vec3(0, 1,  0); }

    glm::mat4 GetViewMatrix() const {
        glm::mat4 rotation    = glm::mat4_cast(glm::conjugate(m_orientation));
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), -m_position);
        return rotation * translation;
    }

    glm::mat4 GetProjectionMatrix() const {
        glm::mat4 proj = glm::perspective(m_fov, m_aspectRatio, m_nearPlane, m_farPlane);
        proj[1][1] *= -1; // Vulkan convention
        return proj;
    }

    glm::mat4 GetViewProjectionMatrix() const {
        return GetProjectionMatrix() * GetViewMatrix();
    }

    // Untuk collision-aware movement
    float GetYawRaw() const { return m_yaw; }

private:
    glm::vec3 m_position;
    glm::quat m_orientation;
    float m_yaw, m_pitch, m_fov, m_nearPlane, m_farPlane, m_aspectRatio;
};

// =============================================================================
// SHADOW AAA — Fase 1 PRD: CSM + PCSS + Temporal Reprojection + Bilateral Blur
//   Modul ini di-inline agar engine tetap satu file, sesuai gaya codebase.
//   Menggantikan single shadow map 4096 + PCF Poisson 25-sample.
//
//   Urutan pass per frame (semua di graphics queue, satu command buffer):
//     0. Depth pre-pass  -> depth + oct-normal + velocity buffer
//     1. Cascade pass x4 -> shadowCascadeArray (D32, 4 layer)
//     2. csm_resolve     -> PCSS blocker search + PCF        -> rg16f
//     3. shadow_temporal -> reprojection + variance clipping -> rgba16f ping-pong
//     4. shadow_blur_h   -> bilateral horizontal
//     5. shadow_blur_v   -> bilateral vertikal               -> shadowFinal rg16f
//
//   Main pass tinggal satu texture fetch di binding = 1.
// =============================================================================

// =============================================================================
// // BLUE NOISE LUT — Void-and-Cluster 128×128, rank[i] ∈ [0..255]
//
// Upgrade dari 64×64: area tile 4× lebih besar → distribusi frekuensi spasial
// lebih kaya, periodicity toroidal lebih sulit terlihat di layar resolusi tinggi.
// Kualitas distribusi meningkat terutama pada shadow penumbra lebar (PCSS >10 tap).
//
//   64×64 (4 KB)  : cocok untuk resolusi ≤1080p, penumbra ≤8 tap
//   128×128 (16 KB): lebih halus untuk ≥1440p dan penumbra lebar
//
// Tile 128×128 bersifat toroidal — sampler REPEAT di GPU tidak menghasilkan seam.
// Diupload sekali ke VkImage R8_UNORM.
//
// Cara pakai di shader:
//   float bn = texture(blueNoiseTex, vec2(px & 127) / 128.0).r;
//   float bn_t = fract(bn + frameIndex * (PHI - 1.0)); // golden-ratio frame shift
//
// Reference: "High-Quality Temporal Supersampling" (Karis, 2014, UE4)
//            "A Gentle Introduction to Blue Noise" (Christoph Peters, 2016)
// =============================================================================
static constexpr uint8_t BLUE_NOISE_128[16384] = {
     41, 211, 154, 139, 248, 185,  14, 120, 127, 252,  21, 162,  93, 237, 105, 153,   8, 119,  61, 186, 156, 248, 101,  36,  73, 250, 181, 132, 214,  38, 228, 184, 142,  47, 157, 103, 115, 214,  77, 170, 206,  38, 184,  91,  75, 197, 136,   8, 208, 233, 159,  12, 128, 172, 210,  47, 113,  68, 217, 107, 177, 146,  21,  47,  83, 237, 187,  87, 252,  17, 107,  94,  22,  51, 148,  15, 111, 192,  36, 247, 188, 216,  66, 175,  82,  49, 237, 167, 137, 222,  49,  95, 151,  22, 172, 155, 252, 104,   8, 208,  60, 242, 120,  66, 172, 241,   2, 212,  43, 146, 225,  58, 112, 196, 177, 253,  61, 112,  12,  99, 207,  90, 113,  60, 191,  44, 126, 115,
     83,   9, 188,  60,  25, 151, 221,  93,  63,  43, 193, 222,   2, 188,  44, 169, 249, 202, 103, 236,  76,   2, 119, 216, 169,   8, 151, 234, 141, 170, 128,  54, 132, 198,  86, 253,  40, 191,   5, 131, 145,  16, 227,  53,  23, 161,  62, 179,  95, 108, 184,  53, 244, 138, 149,  24, 195, 174,  35, 239,  61, 160, 245, 100, 214,  39, 139, 209,  59, 167, 202,  69, 191, 234, 136, 203,  60, 226, 171, 136, 160,  33, 150, 253,  16, 213, 185,  28, 130, 157, 177, 202,  37, 219,  88,  50, 189,  70, 114, 166,  42, 192, 108, 224,  46, 150, 137,  77, 253, 180,  14, 103, 244,  31,  95, 120, 126, 216, 187, 234, 171,   8, 238, 144, 173, 217,  15, 252,
    176, 101, 233,  79, 135, 200,  39, 242, 178, 113,  99, 122,  83, 116, 213,  30,  88,  55,  24, 167, 147,  42,  92,  54, 202, 110,  93,  55,  13, 206,  74, 246,   2, 220,  69,  22, 169, 138, 157, 224, 178,  81, 101, 154, 246, 131, 231, 144,  49,  20,  76, 219,  33, 186, 228,  77, 253,  96, 155,  84,   2, 193,  71, 111, 177, 152,  19, 131, 144,  36, 227,   2, 154, 129,  41, 175,  88,   5, 147,  22, 233,  96,  54, 196, 136,  39, 128, 148, 246,  11,  69, 108, 249,  65, 110, 210,  25, 127,  36, 222, 146,  16,  87,  29, 185, 205, 130,  32,  63, 165,  85, 121,  44, 215,  69,   8, 106,  88,  56,  28, 151,  65, 211,  22, 158,  75, 107, 202,
     48, 113,  21, 213, 169, 130,  67, 160, 211,  13, 230,  56,  35, 245,  70, 111, 122, 180, 226,  69, 214, 190, 230, 154,  80,  33, 223, 192,  86, 156,  29, 140, 161, 179, 105, 146, 228, 206,  35,  61, 241,  44, 212, 138, 188,   3, 211,  30, 194, 253, 123, 113,  93,  63,  13, 160, 136,  50,  18, 229, 141, 212,  31, 123,   7, 243, 189,  66, 235, 157,  90, 139, 183, 218,  26, 253, 159, 103,  70, 204,  82, 111,   4, 156, 129, 144, 225,  60, 198,  84, 217,  19, 118, 168,   7, 232, 118,  92, 242,  77, 199, 173, 251, 161,  94,  18, 226, 142, 196, 235,  29, 203, 163, 145, 184, 240,  48, 229, 165, 138, 251,  86, 182,  97,  51, 239,  26, 122,
     72, 197, 159,  54, 240,   4, 141,  28,  90,  71, 154, 175, 201, 161,  13, 191, 234,   6, 125,  95,  18, 138, 175,  10, 244, 184, 161,  45, 252, 131, 182, 215,  66,  32, 243,  51,   9,  86, 102, 115,  13, 198, 163,  64,  40, 107,  77, 152, 166,  82, 203,   4, 168, 211, 132,  40, 181, 220, 199, 132, 163,  89,  50, 228,  96,  77, 163, 216,  24, 179,  46, 248,  56,  75, 144, 212,  49, 193, 240,  43, 180, 216, 169, 243, 207, 178,   7, 168,  37, 102, 184,  51, 124, 197,  77, 175,  55, 204, 123,   4,  98,  48,  69, 113, 236,  74, 174, 158,  11,  94, 113,  56, 233,  83,  27, 154, 206,  21,  75, 197,  46, 112,  34, 226, 117, 194,  93, 226,
      5, 247,  37, 147,  83, 186, 207, 233, 149, 192, 250,  21, 137, 146, 217, 100,  82,  47, 210, 110,  36, 253, 132,  58, 116, 102,  16,  70, 143,   6, 228,  44, 115,  95, 201, 155, 187,  65, 247, 122, 127,  96,  22, 220, 174, 238,  97,  53, 228, 109,  61,  41, 242, 140, 193, 234,  79, 144,  59,  34, 187, 248, 114, 180, 204,  57,  37, 117, 101,  80, 204,  17, 130, 170,   9,  85, 112,  24, 167, 143,  19,  64,  88,  28,  55,  78, 239,  91, 114, 226, 160, 237,  93,  36, 244, 101,  17, 162,  61, 111, 230, 149, 206,  12,  55, 104,  39, 248,  52, 123, 217, 181,   2, 107, 198, 136, 174, 130, 148, 221,   2, 167, 202, 124,   9,  59, 170, 153,
     88, 115, 182, 225, 137,  32,  77, 171,  53,   6,  96,  62, 224,  42,  68,  31, 173, 247,  66, 120, 203, 158, 144,  27,  72, 235, 208, 135, 197, 166,  82, 101, 191,  11,  77, 234, 135, 173,  29, 216,  73, 112, 253,  88, 149,  18, 203, 179,   9, 120, 218, 180, 154,  70,  28, 106,   6, 170, 129, 235,  69,   8,  81,  27, 126, 120, 252, 192,   5, 240, 146, 135, 225, 152, 244, 197, 121,  75, 208, 133, 250, 152, 227, 116, 124, 105,  22, 203, 123,  10,  79,  29, 110, 214, 152,  44, 222, 181, 250,  35, 184,  23, 132, 177, 220, 191, 116, 205,  79, 106,  22, 152,  71, 252,  42,  67,  10, 244,  53, 181, 100,  67, 240,  80, 104, 250,  31, 214,
     50, 104,  63,  11, 130, 253,  45, 100, 114, 214, 123, 112,  86, 178, 244, 135, 157, 192,  16,  87,  52, 178, 224, 197, 167,  91,  23, 153, 238,  61,  24, 245, 149, 220, 132,  60,  18, 223,  91, 191,   2,  54, 183,  34,  69, 137,  45, 249,  72,  29,  90, 105,  16, 220,  92,  57, 247, 202, 156,  21, 136, 148, 218, 102, 231,  12,  86, 110,  54, 156,  65, 194,  28, 185,  40,  99,  56, 225,   2, 128,  48, 182,   9, 200,  35, 220, 183,  47, 249,  57, 174, 207,  64,   4, 186, 118,  71, 107,  86, 120, 212, 164, 246, 141, 155,  31,  95,   7, 183, 239,  61, 228, 166,  94, 209, 151, 218,  87, 140,  29, 230, 121,  23, 188, 162,  67, 140, 184,
     21, 202, 168, 216, 154, 140, 196,  22, 241, 178,  37, 238,  24, 204, 151,   4, 129, 141, 221, 107, 237,   3,  79,  47, 108, 219,  53, 179,  33, 110, 204, 174,  52, 137,  37, 168, 151, 140,  40, 164, 146, 228, 156, 205, 236, 165, 130, 158, 140, 233, 187,  48, 251, 144, 165, 186, 135,  37,  78, 211, 179, 162, 195,  67,  46, 171, 213,  33, 229, 176, 219,  50,  92, 109,  68, 234,  29, 179, 141, 233, 163,  73,  93, 110, 236,  64, 127, 118,  86, 195, 150, 137, 253, 163,  89, 239,  27, 216,   7,  54,  97,  68,  45,   2,  80, 242,  60, 222, 163, 144,  35, 192, 117,  17, 173,  78,  32, 189, 165, 208,  83, 109,  49, 219,  13, 210, 130, 234,
    158, 245,  78,  24, 184,  58, 220, 161,  79,  60, 195, 125, 103,  74,  49, 231, 209,  27, 167,  40, 125,  96, 117, 249,  14, 186, 120,  76, 224, 123,  89,   8, 159, 236, 186, 206, 251,  54, 197, 243,  85,  27, 133, 128,   8, 191, 222,  15, 200, 149, 166,  75, 194,  33, 210,  13, 149, 223,  98,  56, 253,  39,  13, 243, 111, 199, 145, 161,  71,  96,  20, 118, 238,   5, 164, 207, 130, 157,  78, 197,  38, 219, 190,  46, 154,  84,   3, 101,  34, 230,  14, 131, 192,  23, 105,  52, 197, 145, 161, 236, 194, 118, 226, 100, 211, 168, 146, 133,  23,  92, 210,  82, 103,  57, 226, 142, 249,  51, 133,  11,  61, 253, 177,  91, 149,  41, 174,  72,
     95,  39, 122, 100, 233,  85,   2, 102, 148,  15, 115, 221,   9, 188, 113, 169,  83,  59, 250,  73, 211, 182,  34, 205, 160,  86, 241,   2, 101,  42, 253,  71, 215,  27,  85,  13, 110,  75,   9, 102,  65, 211, 180, 142,  62,  41,  87, 105,  33,  58, 215,   2, 115, 100,  81, 243,  65, 181,   2, 119, 105,  88, 122, 126,  95,  24,  79, 249,   8, 112, 210, 126,  77, 199, 147, 136,  12, 252,  59,  18, 133, 144,  15, 253, 172, 211, 190, 243, 169, 144,  62, 220,  76,  43, 231, 169,  74, 134, 181,  22,  80,  37, 124, 189,  54,  19, 198,  72, 252,  53, 175,   5, 246,  38, 185, 132,   5, 128, 235, 143, 190, 154, 137,  27, 244, 198, 135,   2,
    186, 213, 114,  52, 173,  37, 190, 244, 170, 230,  97,  57, 161, 249,  96,  38, 198, 146, 188,  14, 113, 235,  76, 148,  62,  27, 115, 211, 168, 188, 126, 113,  99,  63, 117, 124,  95, 215, 178, 150, 233,  45,  18, 250, 152, 170, 240, 121,  79, 246,  97, 124, 236,  53, 120,  40, 104, 163, 238,  33, 192, 226,  74,  48, 214, 187,  52, 138, 203, 182,  44, 247,  35, 174, 230,  48, 186, 215, 149, 128, 242, 161,  65, 100,  26,  51, 108,  20,  71, 204,  38, 175,  95, 150, 208, 131,  10, 247,  47, 208, 111, 253,  15, 109, 127, 237,  38, 180, 156, 138, 223, 152, 199, 136, 160,  65, 197, 155, 175, 217,  24,  45, 226,  75, 107,  54, 229, 145,
    241,  70,  12, 250, 126, 215, 114,  63,  39, 205,  26,  82, 201,  31,  70, 234,  19, 103, 161,  91,  61, 122,  10, 177, 229, 103, 191,  48,  87,  17, 220,  32, 202, 178, 242,  49,  32, 238, 135,  24, 165, 110,  92, 197, 217,  74,  25, 209, 127, 113,  46,  22, 205, 159, 180, 227,  22, 202,  69, 215, 152,  18, 184, 239,   3, 153, 234, 168,  36,  98,  62, 121, 107,  19,  72, 105,  88,  40, 136, 201,  29, 178, 227, 116,  78, 232, 126, 119, 223,  97, 157, 247,   2, 183, 139,  35, 223, 150,  70,  99, 155, 175, 201,  69,  94, 118,  87, 211,  16, 128,  43,  76,  22, 129, 239,  19, 223,  34,  58,  80, 101, 207, 165, 185,  11,  89, 167,  30,
     48, 163, 203,  87, 103,  22, 122,  99,  77, 179, 140, 240, 146, 175, 133, 141, 213,  52, 240, 206,  43, 193,  98, 215,  45, 163, 247, 123,  68, 236, 120,  51, 161,   3, 151, 197, 172, 146,  62, 206, 188,  73, 123,  52,   5, 101, 189,  54,  10, 229, 197, 172,  85,  28, 213, 111,  89, 125, 117,  45, 170,  59, 120, 109, 172,  63,  87,  14, 220, 152, 231,  84, 189, 221, 119, 246,  23, 176, 158, 235,  55,  86,   6, 208, 167, 195,  61,  92,  48, 193,  24, 135,  54, 237,  68, 194,  90, 179,  20, 238,  32,  87, 232,  43, 215,   4, 125,  49, 242, 194, 135, 236, 170, 148, 207,  78, 110,  94, 243, 114, 126,   3,  63, 239, 149, 218, 194,  82,
    136, 149, 183,  32,  59, 195, 226,  13, 253, 153, 131,  15,  51, 222,   3, 157, 181,  77,   8, 138, 170, 251,  22, 112,  75,   6,  92,  30, 200, 127, 108,  84, 249, 210, 105,  73, 229,   7,  92,  46, 221,  14, 244, 127, 115, 232, 120, 173,  89, 156,  72, 110, 248, 124,  74,  50, 251,   7,  80, 110, 244,  93, 207,  30,  98, 223, 122, 113,  76, 199,  26, 163,   3,  57, 127, 202,  64, 226,   2,  74, 121, 190, 104, 124,  37,  16, 250, 174,   7, 234, 146,  79, 214, 129,  17, 162,  55, 107, 212, 164, 139,  62,  10, 158, 173, 249, 111,  68, 166, 147,  11, 212,  57,  33, 180,  50, 123, 188,  13,  47, 219, 118,  93,  34, 135,  58,  19, 130,
    235,   8, 219, 112, 233, 172, 157,  52, 187,  34, 225, 168, 128, 189,  63, 253,  38,  96, 148, 218,  30,  80, 126, 199, 234, 120, 173, 221, 101,  11,  60, 188,  20,  55,  89,  24, 137, 161, 253, 112, 121,  99,  40, 185,  88,  58,  27, 253, 215,  35, 184,   7,  56, 102,  20, 191, 151, 172, 224,  27, 187,  11,  77, 252,  47, 198,  22, 246, 184,  52, 143, 252, 209, 102,  37,  92, 124, 194,  96, 111,  32, 223,  46, 242, 115, 101,  80, 215, 139, 130, 164,  30, 173, 151, 137, 206, 252,   6,  85,  51, 197, 219, 128, 143, 193,  56,  96, 229,  35,  84, 186,  71, 114, 253,  89,   3, 236, 205, 158, 174, 251,  77, 180, 201, 162, 248, 144, 211,
    167,  81,  44, 122,  69,   5,  86, 107, 216,  88,  61, 199, 138,  29,  93, 202, 111, 226, 164, 189,  67, 121,  94, 169,  32,  64, 192,  50, 117, 240, 214, 169, 146, 136, 224,  43, 214, 184,  68,  31, 174, 227,  76, 213,  18, 199, 106, 125, 113,  61, 237, 121, 207, 233, 158, 221, 140,  37, 201,  99, 125, 228, 166, 140, 149, 178,  93,  61, 107,   9, 175,  88,  66, 157, 242,  11, 116,  44, 171, 250, 205,  92, 160,  64, 217, 185,  51, 153,  38, 189,  58, 250, 195,  42, 228,  29, 132, 149, 233, 183,  23, 153, 133, 235,  18, 119,  28, 176, 210, 105, 231,  29,  97, 125, 215, 106, 118,  82,  35, 146,  59,  20, 104, 230,  14,  79, 178,  34,
     61, 106, 252,  92, 126, 244, 203,  29, 126, 112,   9, 248, 155, 226,  75, 121,  10,  60,  33, 245, 105,   2, 239,  56, 217, 107, 253,  20,  78, 159,  39,  90, 232, 191, 158, 131, 148,  13, 235,  83, 201,   2, 151, 138, 168, 238,  67,  84,  11, 200,  98, 127,  31,  87, 176,  11,  72, 243,  54, 118,  72,  38, 195,  60,   6, 216,  35, 126, 205, 236, 119,  29, 191, 142, 173, 215,  76, 236,  15,  63, 147,  23, 180,   9,  87, 163, 204,  15, 238,  75, 219,   5,  84, 106,  74, 186,  60, 171,  35, 113,  71, 249, 172,  40,  76, 223, 200, 125,  77,   2, 157, 175, 201,  16,  63, 186,  24,  66, 230, 135, 187, 209,  42, 122, 111,  48,  98, 202,
    151,  17, 198, 175,  22, 119,  60, 179, 240, 121,  77,  42, 132, 175,  20, 243, 127, 198,  91, 116,  53, 214, 181,  23,  84, 123, 127,  98, 181, 206,   6, 112,  62,  28, 245, 176, 204,  56, 118, 104, 159,  65, 249,  46, 131, 147,  38, 212, 172, 155,  48,  73, 113, 195,  63, 118, 106,  89, 184,   2, 215, 111,  96, 237,  83, 162, 242, 116,  83,  41, 101, 214,  56, 231,  25,  51, 105, 201, 163, 140, 188,  77, 245, 113, 228,  27, 134, 143, 172,  94, 121, 112, 126, 239,  11, 160, 142, 203,  79, 222, 103,  13, 200,  89, 105, 115,  61,  22, 241, 121,  59, 245, 119,  47, 226, 167, 249,  99, 213,  14, 130, 235,  88,  67, 190, 218,   2, 241,
    173, 137,  39, 214,  53, 223, 102,  81,  17, 202,  99, 216, 143, 194,  48, 117,  70, 180, 234,  14, 172, 147, 131, 158, 203,   7,  70, 232,  45, 147, 247,  79, 123, 101,   2,  74,  35,  95, 125,  20, 232, 177,  25, 194, 220,   6, 187,  97, 226,  19, 240, 209,   2, 252,  44, 217,  28, 203, 163, 147, 249,  23, 153,  41, 176, 105,  69,  12, 181, 226,  74, 166,   7, 108, 120, 189, 126,  85,  27, 229,  42, 212, 100, 122,  45,  79, 248,  64, 214,  25,  43, 208,  55,  96, 197, 219,  45, 245,   3, 123,  48, 149,  57, 217,   4, 251, 187,  93, 109,  42, 206,  89, 109,  79, 156, 139,  42, 150, 177,  50, 165, 149,  10, 169, 253, 125, 117,  85,
    193, 230, 157,  78, 112, 191,  37, 164, 227,  66,  33, 170,   2, 229, 107,  92, 220,  21, 122,  80, 204,  36, 228, 141, 245,  53, 118, 195,  15, 136, 171, 196,  53, 213, 127, 113, 220, 250, 186, 206,  39,  92, 111,  80,  54, 158, 250,  60, 117,  81, 183, 161, 122,  99, 183, 154, 140, 238,  19,  82,  56, 172, 208, 226,  21, 196,  51, 220, 159,  19, 123, 250, 196,  93,  69, 247,   4, 217,  57, 108, 169,   7,  62, 127, 209, 176, 150,   3, 102, 185, 246, 155, 178,  34, 120,  19, 113, 100, 180,  88, 190, 240, 161, 180, 140, 152,  50, 167, 227, 182,  13, 162,  36, 237,   5, 197,  90,  22, 200,  81, 246, 138,  34, 207,  53, 106,  71,  29,
     46,  66,  12, 248,  91,   2, 242, 134, 144, 185, 156, 240,  88,  68,  25, 202,  56, 169,  45, 113, 252, 103,  62, 190,  33, 110,  88, 167, 217,  68, 224,  21, 126, 241,  47,  90, 121,  10,  80,  58, 119, 213, 126, 231, 179,  89, 108,  23, 124,  36, 103,  45, 224,  85,  14, 229,  77,  47, 132, 223, 191,  93,  66, 117, 126, 253, 123,  93, 200, 104,  59, 113,  43, 222, 172,  35,  98, 115, 124, 252,  88, 191, 233,  19,  94,  33, 192, 233,  49, 116,  75,  17, 225,  68, 253,  81, 125,  62, 236,  24, 213, 110,  35,  64, 234,  31, 209,  10,  79, 142, 219,  69, 193, 174, 104, 223,  69, 242, 136,   3,  58, 220, 185, 152,  93,  19, 236, 211,
    128, 134, 148, 167, 184, 142,  70, 216,  51,  12, 103,  57, 190, 149, 253, 138, 153, 227,  97, 186,  29,  75,   5,  91, 174, 212, 239,  26, 144,  47,  86, 118, 108,  14, 182, 202,  65, 176, 157, 105, 240,   7,  71,  20, 117,  45, 216, 168, 229, 199, 248,  67, 115,  36, 165,  58, 195, 176, 151,  37, 139,   9, 107,  44,  89,   2, 112,  33,  71, 232,  25, 183,  83,  22, 146, 203, 235,  44, 193,  67,  30, 119, 125, 109, 219,  71, 138, 161, 210,  95, 169, 201, 109, 127, 103, 185, 225,  35, 163, 119,  80,  15,  98, 117, 196,  84, 113, 245, 158,  54, 134, 251, 145,  19,  57, 118,  30, 152, 166, 209, 113,  97,  71, 239,  45, 189, 162, 143,
    185, 235, 202,  34,  56, 207, 158,  26, 197, 249, 116, 221,  20, 134, 173,  34, 189,   5,  69, 237, 151, 200, 224, 160,  49,  15,  79, 103, 187, 159, 250,  31,  94, 229, 164,  35, 236,  25, 223,  42, 171, 190, 101, 244, 201,  30, 186,  69,   4,  91, 170,  17, 181, 203, 246, 120, 101,   4, 253, 201, 166, 239, 205, 179, 228,  73, 187, 240, 173, 144, 212, 162, 236, 137,  62, 159,  73, 168,  10, 215, 101, 241,  46, 198,  54, 253, 130,  29,  67,  11, 243,  42, 122,   2,  56,  28, 207,  90, 197,  44, 173, 253, 204,  73,   7, 166,  99,  63, 200,  18, 171,  29, 130, 233,  83, 216,  97, 184, 233,  41, 123,  28, 175,   8, 137, 218, 131,   4,
    169,  22,  78, 108, 232,  16,  82, 104, 171,  92,  77,  43, 163, 210,  55,  84, 243, 108, 123,  22, 170, 139, 128, 134, 249, 148, 204,  58, 234,   3, 199, 173, 209,  61,  83, 151, 112,  97, 194, 147,  84,  33, 123,  62, 165,  96, 235, 141, 155,  49, 107, 216, 151,  72, 110,  27, 213,  68,  92,  60,  24,  77, 142,  18, 164, 209, 153,  56,  10,  89,  44, 132,   3, 207, 186,  17,  88, 228, 139, 152, 185,  76,   2,  87, 178,  16, 143, 173, 227, 190, 126,  72,  92, 210, 241, 171, 151,   7, 105, 228,  66, 154, 139, 223,  46, 230, 192,  36, 116, 216,  85,  48, 201, 156, 178,  43, 121,  17,  64, 108,  86, 250, 143, 205,  80, 155,  58, 249,
     93,  51, 221, 125,  96, 174, 245,  41, 148, 231,   6, 202, 131, 238,  10, 114, 212,  38,  87, 217,  49, 231,  18, 193,  66, 130, 177,  36, 114,  98,  68, 153,  38, 145,   8, 253, 204,  70,   2,  57, 248, 203, 113, 223,   8, 153,  53,  80, 193, 242,  31, 119, 237,   6,  88,  48, 236, 173, 117, 108, 230, 158,  50, 249,  65,  30,  95, 218, 161, 199,  69, 245, 154,  55,  97, 243, 108,  32, 176,  48,  24, 225, 165, 146, 230, 158, 214,  52,  84, 116, 104,  21, 232, 160, 142,  70, 136, 247,  53, 114,  11, 187,  29, 132, 179, 149,  17, 123, 235,  97, 182, 228, 138,  63,   9, 194, 253,  79, 222, 196,  11, 181,  62, 164,  21, 228,  34, 107,
    208, 122, 113,   9,  69, 212, 137, 190,  28,  66, 184, 140, 154,  73, 100, 182,  64, 165, 194, 114, 100,  76, 179,  42, 215,   7, 158, 225,  81, 126,  45, 240, 135, 226, 189, 101,  44, 162, 229, 139, 157,  14,  46,  79, 182, 253,  21, 137, 218, 163,  73, 100,  56, 166, 189, 155, 131, 197,  20,  41, 208, 134, 193,  98, 116, 124, 110,  23, 250, 130, 148,  33, 178, 227,  36, 122, 193, 213,  72, 249, 117, 106, 207,  97,  29,  61, 195,   6, 247,  36, 220, 152,  50, 186,  13,  39, 192,  78, 180, 211,  86, 242, 144,  57, 128, 247,  87,  67, 108,   7,  71, 152,  21, 244,  91, 105, 166, 115,  46, 163, 149, 218,  37, 236,  92, 112, 199,  74,
    239,  30, 191, 253,  42, 160,  55, 131, 217, 165, 128, 251,  34, 193,  25, 235, 149,  13, 240,  67,   2, 250, 144, 163,  92, 241, 140, 198,  21, 119, 218, 185,  28, 131,  64,  22, 180,  78, 209, 130, 184, 234, 103, 213, 145, 132, 204,  37,  95,   8, 182, 201,  25, 210, 229, 143,   9, 138, 224,  82, 177, 147,   4,  40, 204, 237,  78,  47, 139, 185,  17, 214,  79, 117,  70, 127,   5,  55, 163,  90, 201,  56,  12, 246,  75, 113, 122, 127,  94, 179,  60, 167, 205,  84, 109, 237, 214, 163,  19, 119,  43, 164, 198,  20, 217, 161,  42, 208, 172, 251,  40, 207, 131, 160, 213,  30, 229,   2,  95, 208,  28,  77, 116, 192,  53, 158, 180,  10,
    167,  60,  84, 154, 141, 199,   3, 241,  79,  19,  47, 222,  91,  58, 217, 129,  45, 134, 174, 212,  41, 134, 197,  23,  58, 133,  31,  65, 253, 108,  12, 166,  74, 156, 218, 146, 242, 135,  16,  49,  87,  64, 166,  22,  58,  85, 173, 152, 246,  50, 122, 239, 125,  92,  45,  68, 245, 165,  56, 101, 247,  60, 232,  84, 172,  14, 191, 225, 170,  66, 234,  91, 109,  12, 208, 253, 102, 119, 230,  16, 124,  82, 190, 169,  39, 234, 101, 221,  20, 202,  80,   4, 253,  27, 125,  57,  97,  32, 125, 225, 100,  70, 231,  83, 136,   3, 186, 143,  53, 154, 135,  86, 184,  43,  72, 144,  60, 187, 239,  54, 110, 247, 125, 103,  13, 251,  44, 147,
    226, 106, 218, 181,  20, 228,  90, 170, 151, 139, 179, 132,   2, 171, 137, 156, 207,  80,  27, 141, 129, 154, 218,  79, 229, 176, 145, 188,  98,  50, 205,  88, 248,   3, 193,  86,  32, 172, 154, 250,  28, 218, 148, 198, 238,   3, 226,  69, 190, 114,  85,  59, 105,  15, 118, 183,  31, 202, 152,  13,  37, 167, 104, 211, 155,  57, 146,  89,   2, 156,  51, 200, 124, 173,  59, 186,  82,  28, 178, 112, 242,  46, 156, 137, 210,  83,  27,  66, 114, 124, 234, 111, 121,  95, 218, 177, 116,  74, 252, 113,   7, 182,  34, 149, 175,  72, 238,  32, 228, 193,  23, 222,   4, 240, 168, 200,  88, 153, 137, 176,  87,   6,  61, 222,  82, 207, 118,  89,
    194,   7,  40,  72, 102, 116,  67, 194,  51, 237, 208, 157, 198,  78, 243,  15, 186, 253, 161, 194, 237, 178,  12, 109,  38, 204, 161,   2, 224, 171, 150,  59, 141, 174,  53, 104, 233,  60, 200, 131, 138, 180,  91,  34, 105, 162,  49, 100,  16, 233,  35, 214, 168, 252,  81, 220, 122,  73, 235, 181, 217, 120,  72,  20, 138, 253,  34, 215, 107, 244, 100,  20, 222,  41, 233, 153,  45, 204, 222,  64, 100,  30, 220, 129,   5, 177, 193, 249,  43,  97, 186,  35,  64, 197,  48, 234,   2, 195,  41, 205,  61, 220, 131, 249,  48, 214, 106, 118,  98,  71, 169, 140,  61, 128, 134,  12, 223,  41,  18, 213, 161, 201, 145,  37, 171, 111,  18,  57,
    150, 137, 247, 160, 213,  27, 249,  36, 100,  11,  75,  33, 233, 147,  62,  37, 100,  69,  50,   8,  83,  58,  96, 241, 119,  63, 245,  75,  91,  33, 237,  19, 133, 228,  36, 211, 116,  12,  76, 224,   6,  50, 244,  71, 116, 187, 217, 120, 127, 204, 177,   9, 142, 196,  39, 111,  96,  24, 116,  88, 108, 195,  34, 224, 188, 130,  76, 160, 187,  40, 117, 191,  81, 120,  93,   9, 165, 140,  86,   3, 172, 198, 144, 253,  63, 147, 134, 162, 217,  17,  73, 212, 156, 169,  25, 122,  83, 153, 169, 103,  86, 166, 141,  12, 195, 162,  24,  79,   8, 212,  49, 248, 197, 148, 233,  54, 171, 131, 243,  76,  27, 236, 187, 154, 242,  71, 182, 236,
     31, 176,  92, 200,  56, 150, 177, 114, 222, 126, 119,  97,  50, 135, 176, 226, 121, 112, 216, 102, 230,  29, 152, 190,  20, 102,  43, 124, 113, 209, 182, 129, 160, 198,  72, 123, 179,  95, 189, 161, 145, 210, 168, 123,  12, 251,  25,  88,  38, 110,  73,  98, 230, 155,  61,   6, 241, 189,  45,  67,   2, 249,  99,  54, 153,   7, 206,  61,  19, 228,  71, 159, 250,  28, 111, 213, 246,  54, 191, 154, 238,  69,  17, 170,  47, 205,  14, 128,  53, 168, 245, 145,  14, 241,  92, 109, 208, 240,  14,  51, 245,  22, 209,  75, 101,  55, 147, 253, 188, 134, 153,  88,  37, 181,  28, 205,  71, 141, 189,  49,  96, 110,  55,  10,  93,  27, 213, 102,
     69, 217,  46,  15,  82, 230,   9,  85, 202,  59, 253,  25, 190, 219,   7,  93, 200,  22, 178, 159, 141, 206, 168,  72, 214, 127, 223, 192,  11,  62, 142,  38, 242,   8, 102,  25, 253,  55,  36, 232,  65,  89,  41, 203,  96,  61, 164, 209,  67, 247, 125,  51,  27, 181, 135, 216, 167, 139, 209, 229, 164, 143, 176,  80, 238, 133, 178, 246, 122,  89, 180,   5,  53, 196, 170,  66, 100,  25, 225, 131,  49, 139,  91, 110, 222,  82, 239, 152, 139, 198,  83, 107,  69, 215, 188,  58,  35,  71, 185, 144, 194, 132, 156,  39, 242, 171, 218, 139, 163,  41, 235,  12, 115,  76, 155,  90, 253,  15, 156, 230, 119, 125,  79, 220, 198, 143,  45, 162,
      2, 238, 133, 142, 165, 193, 103, 124,  42, 179, 109, 209, 164, 151,  81, 246,  55, 149, 224,  47,  70,  38, 252,   7, 121,  88,  24, 108, 251, 164, 219,  77, 187,  92, 222, 127, 109, 215, 125, 100,  21, 177, 237, 110,  36, 223, 139, 180,  13, 118, 185, 221,  88, 245,  72, 148,  51,  18, 130, 147,  25,  49, 213,  16, 141, 164,  29,  95, 111,  46, 216, 143, 102, 224, 147,  38, 187,  78, 145,  12, 179, 202, 232,  40, 187, 102,  27, 178, 232,   5,  36, 183,  48, 114,   8, 148, 219, 137, 129, 227,  33,  78, 232, 184,  91,   2, 199,  30,  69, 221, 175, 108, 209, 226,   2, 185, 135,  55, 208, 178,   3,  40, 248, 156, 133, 174, 253,  87,
    179, 129, 156, 186,  62, 245,  30,  78, 237,   4,  91,  70,  18,  58, 139,  32, 186,  75,   5, 239,  93, 198, 115, 103,  51, 239, 179,  78,  48,  98,  15, 151,  46, 124,  61, 197,  85,   5, 115, 246, 195, 118,   2,  76, 187, 132, 151, 238,  47, 103, 202,   2, 110, 160,  15, 195,  80, 253, 178, 199,  78, 243,  96, 187,  38, 219,  56, 226,  11, 200, 166, 241,  65,  85,   8, 243, 131, 211, 167, 251,  36, 151,  74, 164,   2,  66, 203,  46, 117,  92, 226, 153, 253,  89, 164, 237, 174,  26, 157,  61, 171, 136,  13,  62, 111, 123,  82, 115, 101,  17,  84,  55, 160,  99,  47, 236, 128, 146,  30,  68, 103, 205, 180,  30,  60,  20, 106, 206,
    140,  36, 224,  23, 208,  49, 121, 214, 164, 147, 224, 116, 245, 176, 228, 130, 161, 208,  99, 121, 175,  21,  78, 224, 173,  35, 148, 213, 140, 234, 173, 205, 248, 117,  14, 237,  42, 188,  69,  32, 126,  55, 219, 162, 248,  19,  56,  91, 216,  76,  40, 171,  66, 209,  39, 234,  94, 157,  34,  62, 112, 170, 118,  70, 202, 104,  79, 184, 146,  70,  31, 136,  20, 210, 177, 156, 138,  18,  69, 104,  86, 216,  21, 242, 137, 149, 249,  79, 105,  61, 206, 138,  23, 204,  40,  77,  98, 202, 243,   4, 196, 221, 145, 204, 238,  44, 226,  59, 246, 206, 186,  26, 243,  65, 169,  34, 212, 163, 225,  92, 239, 148,  71, 140, 193, 231,  78,  50,
    243, 192,  88,  73, 114, 127, 107,  15, 192,  60,  38, 197, 107,  44, 204, 144,  20, 251, 114,  28, 216, 145, 191,  59, 158, 202,  70,   2, 188,  27,  89,  59,  32, 105,  78, 176, 147, 163, 209,  96, 227,  83, 105,  28, 137, 206, 174,   6, 163, 140, 252, 149, 224,  98, 117, 174, 109,   4, 211, 100, 223,   9,  42, 236, 155,   2, 253, 135, 162, 235, 131, 151, 193,  43,  95, 230,  50, 195, 227,  55, 118, 189, 109,  52, 196, 128, 168,  29, 220,  11, 166, 128, 177,  63, 229, 124,  17,  47,  73, 106,  88,  52, 158,  32,  97, 182, 153,   9, 160, 130, 146, 135, 201, 144, 191,  77, 133,   8, 188, 113,  45, 168,  13, 213,  92, 161,  10, 150,
     58,  15, 120, 249,   6, 231,  93,  71, 249, 131, 174,  15,  79,  98,   2,  72, 172,  45,  85,  56, 157, 243,   4, 108, 249,  21,  97, 226, 152,  72, 119, 192, 155, 231, 207,  27, 135, 243,  50, 119,  11, 167, 203,  46, 149,  70, 129, 232, 192,  33,  86,  22,  58, 191,  20,  51, 205,  74, 239,  50, 123, 194,  89, 176,  60, 145,  36, 209,  18,  47, 214, 174, 252, 107, 118,  82,  28, 170, 149,   9, 244,  34,  97, 175, 227,  18, 213, 136, 182, 236,  39, 134, 197,   3, 108, 118, 250, 178, 126, 226, 119,  24, 249, 170,  77,  23, 202, 139, 180,  33, 238,  52,   6,  89, 251,  20, 142, 233,  53,  80,  24, 223, 134, 243,  42, 132, 209, 172,
    110, 219, 126, 101, 180,  43, 159, 183,  34, 153, 139, 239, 163, 222, 182, 235, 132, 199, 224, 105, 180,  68, 118,  40,  85, 123, 112, 172,  39, 250, 103, 221,   9, 169,  98,  66, 220,  16,  82, 127, 251, 186,  65, 240, 178, 217,  26,  64, 132, 152, 229, 184, 138, 243,  88, 229, 135, 150, 167, 185,  21, 251, 107,  17, 230, 191,  95, 173,  86, 109,  62,   4,  78,  32,  58, 208, 247, 129, 137, 179, 207, 163,  75,  10, 144,  65,  41, 157, 129, 148,  72, 247, 154,  90, 213,  66, 192,  86, 112,  18, 203, 184,  71, 111, 223, 117, 253,  41,  67, 208,  81, 155, 222, 105,  49, 158, 204,  97, 173, 249, 154, 191,  83,  56, 183,  26, 236,  80,
     28,  96,  70,  31, 205,  63, 218,  19, 228, 200,  53, 211,  30, 150,  57,  26, 140, 152,   8,  38, 232,  93, 211, 183, 220,  48, 237,  60, 202, 162,  25,  53,  91, 117,  43, 192, 143, 184, 106,  29, 115,  40,  97,  10,  84, 134, 158, 247,  48, 208,  15, 130, 166,  43, 154, 182,  15,  60,  35,  99, 119,  66, 214, 160,  79, 112,  53, 244, 120, 198, 239, 114, 163, 225, 186, 160,   5,  71, 224,  24,  87,  58, 238, 192, 131, 253,  90, 204,  23,  53, 188,  15, 223,  31,  49, 164,  10, 219,  62, 241, 100, 122,  43, 207,  11,  53,  84, 173, 227, 100,  25, 190, 170,  73, 215, 182,  41,  70,  14, 141, 206, 130,   2, 165, 138,  69, 143, 197,
    253,  51, 188, 239, 143, 169,  87, 109, 119,  80,   6,  94,  68, 133, 195,  88, 244,  65, 185, 164, 113,  58,  12, 149, 168,  18, 194,  82,   8, 112, 215, 183, 234,  77, 253,   2, 157,  56, 237, 203,  74, 213, 160, 145, 227,  20, 187, 143,  90, 175,  72,  53, 219,   5,  76, 216, 143, 249, 195, 232,  83,  28, 142,  47, 199,  29, 219,   9,  74,  34, 101, 188,  18, 147, 138, 131,  43, 148, 197,  47, 124, 110, 215, 150,  26, 166, 186,  78, 242, 219, 135,  81, 131, 171, 239,  99, 149,  28, 169,  39,  80,   1, 237,  92, 162, 194, 106, 121,   1, 126, 110, 231,  13, 124,  29, 108, 239, 152, 219,  89,  34,  66, 149, 250, 192, 220,   8, 157,
    134, 147,   4, 133, 128,  13, 247,  51, 126, 103, 253, 188, 142, 241,  36, 160,  15, 211,  80, 253,  26, 200, 140, 244,  69,  95, 127, 120, 231,  95,  64, 146,  18, 173, 204, 103, 225,  33, 163,  91,   4, 232, 106, 196,  53,  76, 207,  38,   1, 242, 139, 201, 148, 247, 102,  32, 171,  85, 116,   5, 207, 153, 183, 246,  98, 170, 144, 156, 178, 229,  54,  84, 212, 244,  27, 204, 231, 165,  91, 250, 101,   7,  38, 134,  69, 226,  44, 102,   1, 144, 173, 208,  55, 138, 194,  75, 228, 200, 143, 135, 215, 176, 147,  62, 139, 241,  32, 218,  77, 244,  54, 120,  66, 247, 117,  85,  23, 197, 134,  55, 240, 176, 211,  40,  91, 106,  55, 179,
    208, 168, 231, 198, 155, 212,  33, 193,  72,  38, 173, 157,  22, 207, 175, 224, 106,  44, 126, 101, 219, 161,  88,  35, 198, 115, 248,  45, 174,  37, 195, 131, 159,  51,  28, 118,  68,  85, 196, 141, 175,  67,  27, 172, 253, 100, 166, 108, 221, 155, 129,  26,  80, 190, 114,  67, 204,  47, 103,  64, 167, 227,  39,  77,   4, 234,  58, 206, 133,  22, 150, 170,  43, 113,  95,  57, 182,  12,  68, 117, 209, 177, 234, 159, 202,  11, 123, 115,  65, 159,  37,  20, 253, 154,   5,  36, 110,  50, 183, 253, 156,  55, 199, 229,  17, 175,  59,  98, 202,  42, 182,  92, 199,  45, 176, 224,  60, 166,   7, 186, 128,  19, 133,  72,  15, 116, 235,  33,
    102,  20,  76,  41,  58, 173, 135, 147, 234, 206,  14, 226,  59, 116, 100,  52, 121, 197, 116,   5,  74,  51, 180, 230,  60,   1, 101,  74, 218,  16, 247, 140, 209, 240,  95, 124, 183, 241,  20, 132, 246,  46,  93, 121,   7,  43, 235,  64,  82, 186,  46, 235, 174,  17, 124, 229,   9, 158, 217, 187,  19,  90, 124, 116, 215, 107,  82,  15, 253, 139, 221, 198,  73,   2, 120,  77, 105, 240,  29, 190,  42,  75, 141,  52, 181, 107, 247, 213, 184, 233, 199, 111,  70, 218, 183,  94, 242, 119,   8,  72,  36, 104,  27,  82, 133, 153, 190,  13, 142, 168,  22, 216, 105, 157,   8, 194, 103, 253, 145, 222, 160, 142, 235, 164, 226, 199,  84,  66,
    121, 224,  96, 119, 237,  88, 222,   1, 165,  50, 121, 106, 126,  89,   1,  72, 237,  33,  89, 232, 192, 139,  14, 129, 165, 223,  30, 204, 139, 163,  80,  55,   9, 111,  62, 200,  10, 152,  48, 209, 147, 185, 225, 111, 213, 188, 148,  28, 204,  13, 133, 160,  58, 209,  40,  96, 184, 241,  30, 122, 253, 105,  57,  21, 191,  40, 125,  95, 184,  39,  67, 107, 230, 165, 251, 208,  45, 154, 219, 122,  98, 224,  27, 243,  92,  77,  30,  57,  95,  44,  85, 122, 102,  49, 142, 206,  61,  84, 212, 124, 230, 192, 169, 220,  41, 250,  85, 232, 157, 134, 252,  74,  34, 240,  82, 122,  31,  90,  42,  70,  26, 204,  50, 186, 147,  26, 161, 247,
     28, 190, 166,   9, 105,  25,  64, 185,  82,  98, 240,  73,  29, 215, 248, 179, 153, 210,  59, 160,  28, 243, 154, 201, 134, 145, 185, 156,  53, 132, 188, 231, 172,  83, 229,  40, 220, 167,  96,  76,  28, 161,  16,  87,  61, 160, 129, 138, 168, 248, 146, 220,  91, 108, 251,  76, 118, 127,  82, 112,  45, 202, 235, 174, 154, 243, 113,  48, 204, 154,  10,  91,  30, 194, 146,  22, 138, 172,  84,  54,   1, 171, 132, 155,  14, 207, 149, 175,  12, 153, 244,   7, 228, 173,  18, 155,  32, 180, 102,  20, 116,  96,  10, 146, 203,  68, 112,  30, 211,  62,   6, 144, 174, 206, 114,  67, 214, 164, 201, 109, 244,  99,  10,  87,  63, 135, 181,  54,
    113, 217,  67, 245, 179, 208, 155, 246,  32, 115, 199, 176,  46, 192, 130,  25, 140,  11, 186, 134, 128, 208,  64,  24, 237,  49,  77, 251,   6, 222,  39,  99,  31, 147, 181, 115, 102,  55, 252, 113, 222,  57, 238, 198,  34, 246,  10, 223,  50,  94,  34,  73,   3, 164, 191,  21,  52, 101,  61, 223,   6, 166,  71, 143, 212,  29,  78, 227, 167, 134, 245, 180, 141,  52,  81, 179, 227,  12, 200, 110, 253, 144, 212, 185,  46, 232, 130, 136, 218, 193,  62, 162,  37,  79, 249, 133, 225, 161, 246,  47,  75, 236,  63, 135, 165,   3, 124,  50, 182,  87, 194, 225,  95,  56,  16, 229, 150,   3, 181,  78,  55, 125, 119, 252, 217,   1, 206,  92,
    124,  19, 156,  47, 146,  38, 129, 136, 216,  55,  19, 110, 232, 152, 135, 166, 223,  78, 245, 170, 149,  41,  83,  97, 181,  15, 107,  93, 196, 146,  74, 160, 201, 250,   4,  75,  29, 204, 177,   8, 193, 121,  98,  69, 176, 135, 128, 192,  70, 215, 175, 197, 231,  43, 145, 212, 237,  11, 197, 154, 183,  85, 103,  14,  60, 117, 189,   1,  57, 213, 147,  46, 235, 131, 212,  63,  99,  39, 235,  70, 189,  58,  18, 104,  71, 192, 161, 252,  38, 100, 142, 183, 210,  99, 188,  57,   3,  88, 193, 147, 174, 206,  35, 246, 185,  93, 230, 106, 243, 116,  44, 163,  32, 155, 191, 100,  49, 242, 137, 222,  29, 196, 105,  42, 159, 140, 232,  39,
    250,  96, 206,  85, 227,  74, 195,   7, 165, 147, 251,  91,   6,  76,  57, 203,  40,  95,  51,  22, 226, 188, 253, 114, 211, 168, 219,  33,  65, 176, 243,  15, 108,  53, 157, 189, 231, 145,  66,  91,  45, 127, 110,   1, 212, 152, 232,  21, 158, 102,  18, 140, 128,  62, 179, 130, 140, 170, 229, 133,  31, 247,  47, 224, 176,  91, 247, 124,  98,  23,  74, 192,  20, 162,   7, 248, 149, 166, 116,  92,  27, 163, 223,  89, 123,   5,  58,  90,  19,  73, 224,  51, 116,  22, 221, 151, 140,  67,  37, 222, 132,  13,  86, 152,  42, 212,  21,  74, 127,  14, 101, 238,  76, 139, 249, 176,  86,  22, 130, 154, 174, 238,  84,  19, 187,  82, 171,  70,
    108, 169,   4, 186, 133,  20, 144, 233,  72, 203,  63, 158, 221, 171,  27, 253, 107, 121, 196, 109,  69,   5, 126,  32,  58, 150, 138, 238, 132,  25, 137, 210,  67,  93, 218, 137, 129,  16, 161, 239, 207,  29, 249, 165,  85,  38,  58, 181,  83,  46, 253, 153, 133, 240,  23, 157,  35,  74,  50, 148, 212, 131, 140, 159, 200,  24, 111,  68, 231, 114, 168,  93, 222, 138, 199,  85,  28, 195,  50, 218, 151, 137, 246,  36, 198, 239, 105, 180, 204, 150, 241,  13, 125,  85,  43, 169, 239, 203, 129,  22, 162, 140, 181, 218,  60, 120, 196, 171,  59, 222, 203, 174,   1, 210,  24,  65, 219, 144, 203,  71,  16,  50, 212, 151, 242,  52,  21, 199,
     35,  61, 236, 138,  57, 249, 173,  88,  26, 180,  38, 100, 189, 141,  88, 181,   4, 233, 126, 214,  88, 121, 103, 223,  81,   7, 184,  53, 153, 198, 128, 164, 231,  20,  44, 171, 248,  39, 134, 175,  76, 120,  59, 183, 221, 102, 145, 240, 200, 110, 221, 185,   9, 201,  71, 221, 191, 252, 103,  15,  69, 188,   4, 239,  78,  52, 217, 186,  41, 200, 253,  35,  61, 152,  41, 108, 181, 239,  74,  20, 199, 177,  52, 110, 119,  78,  27, 221,  47, 135, 174,  77, 109, 247, 199, 101,  15, 135, 177, 252,  61, 229,  71,  99,  26, 108,  85, 254,  33, 120,  81,  53, 150, 129, 134, 157,  37, 170,  55, 234, 129, 166, 141,  60, 205, 100, 121, 224,
    191,  88, 153,  27, 207, 158,  48, 212, 104, 124, 218,  13, 242,  44, 212, 156,  61,  93,  37,  18, 184, 237,  44, 195, 166, 247, 101, 206,  37, 233,   1, 187, 131, 143, 197,  80,  59, 210, 150, 225,   6,  96, 229,  15,  72,  24, 162, 137,   4,  36,  78,  55, 165,  41,  91, 109,   2,  87, 204, 172, 232,  95, 152,  40, 100, 122, 166,   7, 151, 103,  14, 118, 179, 243,  71, 219,  16, 124, 105, 227,  97,   3,  83, 211,  15, 228, 154, 130, 165,   6, 197,  33, 217, 160,  27,  76,  50, 217, 147,  45,  91, 190,   8, 245, 159, 227,   5, 124, 186,  95,  24, 193, 245,  40, 230, 194,  98, 254,   5, 188, 134, 227,  12,  91, 175,   5, 127, 114,
    239,  11, 214, 183,  74,  96,   9, 117, 245,  52, 109,  81, 122,  72,  19, 237, 146, 174, 227, 114,  57, 151,  21,  95, 114,  66,  20,  84, 167,  76,  58, 149,  38, 240, 162,   6,  97, 184,  21,  50, 108, 194, 126, 112, 205, 251,  50, 173, 212, 120, 126,  95, 209, 148, 245, 123, 181, 118,  43, 154,  57,  28, 205, 180, 221,  20, 249,  94, 180,  66, 215,  84, 205,   2, 169, 115,  94,  47, 127, 118,  65, 241, 145, 184,  62, 170,  40, 141, 248,  73, 232, 119,  91,  61, 189, 231, 172,  86,   5, 198, 155,  39, 137, 147, 198, 177,  68,  41, 233, 109, 220, 156, 139,  63, 175,  11,  73, 113,  86, 150,  31, 194,  74, 218, 109, 251,  47,  96,
    177,  68, 132, 144,  34, 227, 199,  66,  18, 190, 230, 168, 204, 116, 103, 188,  84,  46, 200,  77, 250, 172, 207, 233,  48, 188, 229, 144, 219, 103, 254,  90, 206,  25, 135, 216, 237, 116,  89, 203, 235,  57,  33, 122,  91, 189, 133, 225,  64, 107, 247,  12, 229, 176,  20,  49,  80, 238,  18, 221, 134, 249,  83,  63, 118, 106, 204,  79,  43, 238, 158,  48, 106,  89, 234,  57, 203, 254,  10, 207,  42, 166,  32, 100, 254,  92, 190, 210,  58, 110,  98,  51, 171,   1, 142, 131, 152, 243,  68, 103, 236, 175, 214,  29,  53,  92, 113, 203, 127,  12,  67, 171,  16,  89, 206, 148, 222,  24, 209, 172,  47, 244, 156,  26,  57, 195,  76,  24,
    138, 222,  47, 252, 171, 129, 150, 177,  85, 126,  36,  67,   5, 246,  53,  29, 210,  14, 102, 163,   2,  90, 120,  80,   9, 159, 134,  30, 181,  10, 156,  48, 172,  68, 128, 153,  63,  34, 174,  71, 151, 168, 245,  75,  19,  42, 154,  14,  88,  27, 196,  73, 115, 105,  63, 192, 210, 101,  73, 184, 145, 166,  19, 126, 236,  57,  33, 145, 196, 133,  20, 232, 175,  36, 148,  19, 159,  75, 176,  85, 233, 190, 154, 216,  23, 150,   8,  86,  31, 184,  16, 223, 194, 254, 128,  22, 192,  34, 117, 208,  18,  62,  98, 115, 222,  14, 245, 122,  49,  89, 250, 201, 112, 235,  33, 136, 167,  62, 236,  99,  69, 137, 130, 181, 141, 224, 167, 150,
    193,  18, 161,  83,   1, 137, 241,  29, 217,  99, 119, 197,  95, 175, 218, 157, 132, 241, 145,  50, 214, 111,  30, 198, 147, 251, 128, 208,  68, 114, 191, 222, 138, 246, 182,  19, 197, 105, 254,  26, 139,  11, 187, 212, 105, 231, 138, 203, 235, 180,  46, 164,  30,  89, 224, 161,  10, 125, 112,  25, 198,  44, 218,  97,   1, 182, 159, 242,   9, 172, 141,  75, 201, 134, 187, 220, 142, 196,  30, 110, 121,  11, 106,  81,  51, 201, 136, 220, 161, 242, 148, 136,  40,  68, 210, 167, 223,  82, 126,  48, 110, 123, 248,  76, 190, 170,  80, 104, 217, 183,  28, 149,  42,  99,  74, 246,  49, 192, 142,  28, 185, 211,  41, 238,  12,  92,  34, 244,
    105,  91, 208,  60, 231, 189,  44,  75, 111, 251,  24, 232,  43, 148,  74, 137,  58, 170, 192,  74, 233,  42, 178,  71, 216,  52, 173,  44,  92, 239,  21,  81,  41,   5, 144, 231,  48, 161, 206,  91, 219, 132,  46,  94, 163, 182,  57,  78, 159, 130, 138, 207, 242, 185,  37, 149, 254,  44, 218,  59, 239,  87, 120, 190, 113,  84, 208,  68,  98, 223,  57, 153,  24, 250,  86,  67,  39, 136, 245,  55,  94, 225,  66, 243, 165, 130, 233,  47,  69, 129, 205,  25, 160, 133, 148,  47, 101,   7, 251, 181, 224,  86,   3,  44, 158,  22, 225,  64,   4, 158, 102,  58, 224, 181,   1, 151, 131,  12,  86, 162, 251,   1, 147,  84, 161, 213, 112,  64,
    234,  41, 181, 135,  22, 128, 164, 210,  12, 176,  63, 167, 110,  18, 194, 254,   7, 223,  34,  96, 168, 143, 244,  99,  19, 132, 141, 222,  27, 104, 148, 206, 164, 129, 215,  75,  94, 111,   3,  66, 177, 155, 227,  66,   6, 248,  31, 145,   9, 252,  58, 148,   1,  71, 136, 198,  67, 117,  93, 173, 126,  16,  69, 254,  37, 224,  24, 120,  42, 191,  90, 209,  51, 161,  10, 101, 235, 166,   1, 210, 180,  33, 197, 146,  29, 185,  18, 173, 139,   6, 177,  76, 244, 186,  12, 239,  64, 122, 199,  70,  38, 171, 196, 232, 130, 145,  38, 194, 116, 241, 204,  80, 160, 140, 196, 217, 172, 230, 202,  65, 106,  56, 172, 202,  71,  47, 187,   4,
     82, 217, 143, 129, 156, 222, 145,  54, 194,  90, 153, 204,  86, 225,  48, 178,  88, 108, 118,  13, 205,  57,  26, 189, 164, 240,   1, 197, 157, 188,  54, 245, 135,  61, 191, 167,  27, 237, 185,  44, 246,  78,  26, 197, 142, 132, 217, 174, 197,  37,  95, 218, 178, 130, 230,  17,  85, 226,   3, 202, 114, 103, 211, 164,  58, 155, 104, 177, 251, 110,  16, 236, 135, 183, 223, 113,  57, 188, 144,  74, 155, 138,  52, 132, 213,  78,  97, 251, 151, 216,  45,  94, 214,  57,  86, 170, 216, 113,  91,  16, 155, 136, 149,  56, 206, 136, 252, 168,  87,  47,  16, 177,  32, 254,  60,  23,  94,  37, 149,  18, 218, 119, 234,  17, 103, 254, 148, 168,
     28, 159,   9, 243, 199,  32,  84, 236, 118,  40, 242,   3,  58, 139, 158,  25, 123, 211,  47, 249, 136, 156,  89, 210,  66, 148,  87,  60, 233,  73,   8, 175,  31, 232,  12, 132,  55, 211, 128, 134,  20, 144, 129, 238, 160,  39,  92,  64, 134, 225,  73,  24, 162,  49, 143, 156, 175,  41, 124, 248,  29,  50, 181,  10,  98, 194, 233,   7,  71, 125,  40, 172, 150,  74,  28, 122, 211,  21,  90, 228,  16, 251, 169,   4, 238,  56, 110,  40, 190,  65, 235, 112, 122,  36, 201, 142,  24,  42, 236, 178, 214, 246,  11,  89, 175,  49,  77,  24, 106, 216, 125, 234, 106,  88, 121, 110,  73, 241, 135, 185,  77,  96,  43, 189, 118,  31, 215, 138,
    240,  72, 188,  52, 104,  71, 182,   7, 108, 124, 216, 103, 179, 236, 209,  73, 239,  63, 195,  80, 182, 228,   6, 108,  40, 227,  33, 177, 109, 121,  98, 200,  84, 142, 153, 251, 173, 140, 151, 229, 163, 214, 187,  46,  72, 180, 233,  20, 164, 142, 185, 106, 246, 202,  29, 239, 212, 104, 185,  62,  81, 229, 146, 242,  81, 143,  52,  95, 214, 116, 225,  64,   4, 248, 191,  43, 104, 243, 159,  63, 102, 208,  83, 190, 159, 137,  12, 223,  84, 103,  22, 171,   1, 102, 227, 159, 188,  73, 123, 103,  26,  66, 117, 234,  31, 214, 156, 231, 180,  66, 117,  53,   6, 207,  42, 194, 167, 209,  45, 157, 247,  25, 127, 217,  59,  87, 180,  55,
     96, 114, 223,  89,  23, 254, 162, 208,  66,  22,  81,  49, 145,  15,  92, 118, 102,   1, 164, 146,  30,  71, 118, 252, 157, 186, 118, 217,  16, 254,  43, 156, 219,  65,  40, 195,  17,  71,  32, 189,  68,  10,  85, 137, 212,   1, 105, 203,  47, 243,  11,  83, 117,  66, 101,  79,  56,  12, 119,  96, 194, 162, 130,  33, 218,  20, 163, 200,  28,  85, 181, 107, 202, 117,  84, 126,  71, 174,  32, 199,  44, 112,  23,  67, 227, 148, 182, 166,  29, 205, 155, 187, 254,  70, 120,  13, 246,  96, 206,  52, 120, 202, 107,  75, 191, 101,   1, 141, 199,  19,  97, 221, 183, 151, 236,  14, 140,  62,   4, 131, 200,  65, 110, 122, 241,  11, 104, 203,
     48, 165,  13, 177, 142, 134,  35,  93, 240, 172, 196, 251, 164, 192,  60,  35, 175, 223,  43, 109, 206, 172,  58, 195,  90,   9, 104,  53,  89, 172, 208,  20, 108, 181, 225,  83, 104, 243, 208,  95,  51, 252, 175,  32, 147, 248,  80, 116,  70, 152, 198,  42, 216,   5, 189, 115, 163, 200, 244,  26, 220,   5, 139, 204,  67, 179, 246, 138, 157,  48, 245,  23,  95,  36, 213, 236,   6, 205, 142, 133, 176, 237,  97, 126,  31, 204, 246,  59, 118, 241,  51, 142, 212,  46,  90, 110,  60, 167,   3, 232,  82,  37, 173,  19, 148, 246,  58,  85,  41, 249,  75, 162, 138,  63,  82, 161, 249, 181, 221, 144, 164, 227,   7,  79, 176, 159, 231,  20,
    193, 249,  63, 211, 156, 227,  55, 185, 146, 136,  39,  89,  30, 109, 205, 248, 143, 189,  95, 246,  21, 231, 104,  43, 219,  73, 239, 162, 196, 141,  80,  59, 244,   6, 125, 112,  48, 161,   1, 115, 124, 108, 221, 158, 132,  54, 183, 230,  23, 175,  94, 235, 165, 148, 254,  23, 228,  87,  50, 173, 150,  74,  43, 170, 134,  38,  87,   2, 222, 192,  77, 165, 228, 179,  55, 112,  97,  61, 250,  11, 155, 215,  58, 121, 113,  50, 105, 124,  90,   8, 109,  74,  25, 168, 233, 202,  31, 221, 137, 185, 161, 254, 215, 137,  65, 169, 207, 145, 166, 131, 204,  30, 232,  22, 214, 105,  33,  93,  77,  21, 189,  46,  94, 208,  35,  64, 134, 152,
     29, 104, 118,  38, 191,   1,  85, 218,  14,  63, 207, 155, 223,  69,  10, 156, 134,  20,  74,  53, 157,  84,  11, 168, 147,  26, 204,  39,  68,  24, 231, 167,  99, 119,  35, 203, 231,  73, 175, 238,  39, 203,  75,  21, 196, 165,  13,  88, 212, 106,  30,  63, 138,  52,  85, 178,  42, 127, 106,  67, 235, 187, 128, 252, 152, 228, 102,  68, 147, 130,  13, 143,  65, 153,  16, 122, 189, 162, 220,  95,  35, 192,  14, 254,  87,   3,  76,  36, 220, 181, 230,  94, 195, 147,  16, 181, 155,  77,  42, 146,  23, 100,  53, 185, 225,  14,  46, 237, 191,   7, 148,  53, 172,  95, 191,  52, 123, 113, 233,  56, 100, 238, 116, 186, 251, 142, 221,  79,
    174,  90, 224, 125,  74, 238, 152, 131, 166, 233,  97,   6, 117, 178, 235,  79,  44, 230, 199, 147, 219, 190, 140, 242, 132, 177, 128, 153, 245, 133, 148, 192,  51, 214,  93, 178,  19, 145, 195,  89,  63,   9, 103, 244,  47, 225, 141, 154,  59, 246, 115, 184, 204,  17, 212, 109, 221, 122,   8, 201, 137,  19, 215,  59,  10, 184, 206,  47, 241, 177, 216,  35, 196, 254,  92, 234,  23,  39,  82,  51, 144, 134,  74, 179, 210, 168, 239, 200, 149,  65,  29, 162, 245, 130,  66,  88, 251, 130, 194, 236,  70, 118,   5, 155,  82, 119, 111,  92,  34, 136, 224,  86, 254, 120,   1, 230, 207,  13, 195, 152, 174,  29, 124,  17,  57, 167,   1, 207,
    242,  56,   9, 113,  98,  26, 205,  45,  72,  25, 188, 245,  59, 106,  94, 211, 173, 128, 164,   5, 101,  60,  32, 201,  49, 227, 139,  19, 210,  83,   1, 224,  29,  72, 251, 157,  56, 135,  34, 224, 169, 152, 182,  91, 121,  72, 204,  41, 188, 162,   3,  75, 230, 151, 168,  27,  76,  96, 250,  33, 148, 131, 158,  83,  33, 121, 112,  22, 159, 135,  56, 101, 113, 172,  51,  76, 197, 137, 206, 175, 242, 157, 226,  23, 150, 135, 128,  20, 171, 133, 208,  53,  11, 139, 217,  37, 144,   8,  61, 168, 208,  90, 240, 205,  31, 232,  70, 177, 215,  61, 157, 183,  35, 110, 127,  67, 166,  81,  40, 254,  64, 219,  75, 110,  89, 196, 130,  40,
    159, 143, 186, 246,  51, 173, 140, 252, 182, 146, 135,  44, 202, 159,  31,  57,  13, 135, 238,  42, 115, 250, 161,  79, 108,   4, 191,  59, 173,  43, 159, 129, 137, 173,   8,  86, 237, 205, 130,  17, 140, 210, 230,  28, 111,   5, 254,  93,  26, 215,  90, 120,  37,  98,  64, 241, 195,  39, 175, 211,  57, 242, 177, 203, 238,  96, 225,  83, 196,  10, 234,  85, 219,   3, 120, 222, 167, 152, 229,   1,  65, 195,  53,  98,  44, 229, 183, 142,  50, 252,  81, 154, 225, 175,  55, 158, 207, 134, 226,  18, 110,  39, 166, 143, 100, 188,  43, 153,  21, 243,  75,  14, 212,  78,  41,  99, 239, 180, 138, 131,   3, 161, 193, 241,  42, 232, 137,  72,
     17, 201,  81,  22, 215, 132, 196,  36,  82, 230, 170,  85,  17, 227, 186, 254, 151, 183, 209,  96,  68, 179,  14, 213,  96,  71, 254, 143, 231,  94, 186, 246, 147, 202, 106,  43, 185, 150,  71, 254,  79, 133,  46,  67, 195, 175, 104, 125, 114, 238,  54, 196, 250, 179,   7, 144, 161,  62, 112, 155,  78,   1,  41, 109,  64, 169,  36,  60, 251, 144, 171,  28,  67, 200,  43, 100,  13,  59, 112, 101, 121,  28, 216, 171,  80,   9,  66, 203,  92,   5, 191, 135,  35,  86, 190, 242,  26, 180, 150,  83, 245,  57, 195,  11,  61, 163, 251, 200, 109,  96, 190, 145, 168, 245, 193, 119,  58,  26, 226, 146, 213, 102,  49, 150,  10, 144, 181, 223,
    101, 231, 170, 153,  66, 162,  15,  99, 115,   3, 208, 153, 108,  73, 141, 130,  66,  22,  81,  30, 200, 149, 135, 233, 183,  39, 130, 164,  30,  73, 208,  57,  17, 230,  67, 118, 221,  12, 173,  51, 193,  22, 165, 240, 142, 221,  52,  18,  72, 176, 158,  14, 107,  78, 213, 134, 222,  11, 234,  99, 193, 222,  92, 119,  20, 198, 147, 179, 131,  38, 193, 135, 157, 248, 109, 187, 238,  84,  25, 252,  87, 126, 113, 248, 191, 145, 240,  26, 224, 166, 144,  66, 238, 104,   1, 120,  95,  48, 199,  35, 155, 178,  96, 236, 210,  27,  78,   9, 125,  38, 234,  57, 130,  29, 218,   9, 106, 201, 159,  20,  87, 180, 225,  96, 167, 205,  27,  62,
    115,  52,  34,  87, 225, 143, 241,  58, 222,  70,  30, 249,  47, 205,   7, 164, 221, 145, 246, 159, 227,  46, 129,  60,  24, 137, 193, 215,  11, 126, 108,  40,  90, 183,  27, 126,  97,  40, 233, 156, 129, 146, 202,  80,  15, 159,  84, 235, 192,  40,  97, 221, 123,  43, 191,  28,  83, 106, 178,  25,  48, 152, 183, 254,  80, 217, 134,   3, 211,  71,  93, 230, 142,  20, 174,  71,  35, 202, 173, 213,  39,  70,  11, 101,  31, 157,  84, 116, 103,  45, 216,  21, 198, 169,  63, 216, 111,  69, 233, 130, 212,  20,  79, 121, 127, 114,  98, 216, 119,  68, 203,   6, 136, 154, 177,  89, 124, 248,  49, 189,  68,  38,  18,  77, 251, 108,  85, 125,
      8, 185, 254, 137,   7,  45, 181, 198, 127, 121,  95, 182, 161,  92, 238,  41, 195,  53, 131, 138,   4, 189, 243, 146, 169, 224, 151,  54,  99, 242, 120, 227, 164, 150, 248,  79, 198, 114,  85, 211,   5, 244,  39, 108, 185,  35, 205, 140, 153, 229,  23, 117,  63, 241, 141, 167, 252,  54, 207,  86, 241, 134,  60,  10, 156,  33, 233, 128, 157, 244,  55,   8, 214,  62,  97, 223, 146, 129, 135, 154, 184, 236, 203, 165,  54, 219, 198,  37, 179, 248,  87, 160, 115,  31, 254,  82,  23, 162, 145,  10, 136, 254,  63, 107,  39, 233, 182,  50, 228, 175,  90, 165, 209,  45, 238,  67,  33, 114,  96, 219, 134, 244, 155, 198,  60,  38, 217, 238,
     79, 210, 151, 130, 201, 167,  79,  23, 107,  40, 228, 144,  61,  24, 110, 176,  99,  14, 212, 167,  69,  92,  19, 206,  77,   6, 247,  83, 114,  35, 193,   8,  66, 204,  46,   2, 224, 124,  28,  67, 176, 133, 216,  95,  63, 250, 128,   3,  56,  79, 209, 170,  88,   1, 153,  72, 197,  33, 147, 163,  16, 211, 144, 196,  96, 168,  56, 189, 140,  34, 177, 107, 164, 198,  44, 158,  16, 245,  51,   5, 128, 149,  47,  94, 242, 109,   3,  70, 147,  59,  13, 107, 211,  50, 124, 176, 203, 241, 187,  60, 169, 196,  26, 220, 171,   1,  89, 154,  16, 109,  27, 248,  78, 147,  17, 190, 223,  81,   7, 167, 145, 206, 129, 140,   3, 169, 189, 120,
     99,  58, 174,  27, 236,  94, 117, 251, 214,  87,  11, 192, 235, 208, 119,  67, 244,  80, 184,  33, 218, 110, 176,  41, 102, 115,  48, 180, 209,  69, 174,  93, 218, 109, 122, 179,  59,  99, 249, 196, 141,  51, 166,  11, 223, 144, 133, 166, 240, 102, 187,  47, 233, 181, 215,  20, 137, 129, 237,  68, 130, 174,  29, 225,  46, 246,  75,  22, 206, 131, 222,  78,  31,  89, 236, 133, 189,  86, 169, 232, 139, 218,  15, 176,  77, 120, 184, 233, 210, 165, 194, 235,  75, 100, 220, 118,   8,  47,  81, 218,  42, 143, 159,  97, 192,  71, 249, 205,  82, 235, 191, 116,  52, 221, 132, 140, 158, 181,  58, 238,  27,  47, 174, 227,  91, 151,  67,  25,
    245,  39, 143, 218, 103,  48,  13,  63, 174,  52, 164, 132,  33,  83,   1, 225, 158,  39, 107, 234, 149,  55, 250, 155, 215, 189,  27, 160, 235,  18, 122, 250,  54,  25,  86, 238, 111, 171,  12, 157,  34, 231, 191,  78, 158,  44, 179, 201,  39, 116,  11, 125, 105,  37,  92,  59, 229, 182,   5, 193,  42, 248,  73, 106, 119, 186, 112,  88, 239, 152,  15, 192, 248, 141,   1, 211,  58,  33, 196,  67,  26, 132, 193,  64, 211,  41,  99, 123,  26,  97,  41, 152,   4, 186,  28,  90, 109, 127, 121,  95, 232,   6,  83, 239,  50, 124, 113,  44, 168,  35,  65, 100,   1, 162, 199,  37, 252,  23, 133, 192,  85, 105,  14, 191,  51, 240, 214, 112,
    165, 134, 194,   1,  70, 180, 225, 154, 137, 200, 239,  68, 149, 168, 188,  49, 145, 205,  87,   8,  75, 195,  29,  88,  64, 232,  92, 118,  80, 127,  46, 100, 185, 161, 197,  15, 218,  52,  80, 212,  93, 106, 115,  27, 206, 234,  15,  87, 220,  68, 254,  82, 205, 114, 243, 166, 146,  79, 134, 220, 139, 154,  91,   1, 216,  37, 125,  10, 174,  46, 143,  59, 132, 177,  73, 166, 148, 129, 215, 156,  79, 248, 146,  30, 161, 252,  12,  79, 115, 243,  69, 178, 141, 245,  70, 228,  56, 250,  34, 108, 165, 199,  56, 121,  28, 224,  18, 101, 199, 142, 155, 214, 176, 233,  85, 104,  74, 211, 150, 229,  63, 121, 251,  73, 110,  33, 176,  12,
    206, 231,  81, 162, 248, 145, 132, 207, 128,   4,  43, 136, 212, 252,  93, 112,  17, 175, 246, 162, 140, 171, 227,   2, 123, 167,  12, 211,  59, 199, 112,   4, 232, 144,  69, 116,  39, 189, 146, 240,  64,   1, 254, 122,  92,  61, 118, 105,  23, 163, 194,  26,  57, 122,   8, 190,  27,  50, 164,  32,  65, 203, 233, 163,  64, 199,  98, 225,  67, 196,  94, 231, 153,  43, 228,  22, 254, 136,   9, 229,  40, 170, 129, 223,  87, 190, 149, 221,  47, 202,  18, 223, 129,  45, 156, 195, 171,  13, 204,  69,  22, 248, 182, 107, 206, 164, 184,  62, 220,  13, 254,  81,  31,  61, 114,  10,  49, 140, 170,   1,  96, 215, 165,  24, 205, 121,  94,  72,
    151,  21,  52,  98,  39, 189,  17,  56, 244, 151, 188,  87,  38,  11,  59, 196, 222,  73, 109,  22, 220,  45, 103, 115, 202,  55, 252, 104,  34, 242, 171, 214,  83,  32, 207, 254,  94, 163, 136,  25, 175, 202,  74,  40, 185, 126, 247,  49, 184, 136, 150, 230, 173, 214,  81, 100, 222, 201, 236,  90, 175,  17,  45, 133, 175, 237,  27, 114, 160, 254,  35, 182,  12, 197, 111,  94, 186,  60, 175, 104,  89, 204,   1,  50, 134,  20,  61, 159, 183, 104,  85, 167, 136, 209,  19, 140, 131, 149, 180, 223, 123, 112,  65,  12,  93,  72, 247, 119,  92, 171,  47, 116, 207, 123, 245, 188, 220, 129, 198,  54, 115,  40, 187, 153,  82, 229, 185,  55,
    131, 139, 174, 213, 113, 230,  82, 167,  33,  72, 228, 174, 141, 159, 231, 104, 121,  32,  52, 202,  91,  68, 186, 235,  95,  37, 182, 149, 164,  89,  24,  64, 157, 102, 176,  61,  28, 229, 206,  49, 132, 143, 162, 221, 107,   6, 208,  79, 227, 128,   3,  91, 109,  31, 250,  54, 138, 152,   8, 105, 254, 148, 195,  78,  12, 143,  53,  86, 209,   3, 110,  78, 213,  86,  56,  32, 210, 145,  20, 190,  57, 243, 137, 184, 235, 143, 207, 245,   3, 121,  57, 254,  37, 182,  66, 218,  78, 237,  44, 100,  83,  36, 214, 159, 231,  31, 126,   3,  55, 229, 104, 193,  16,  98,  38, 163, 146,  17, 247,  81, 234, 123,  12, 244,  45, 100,   4, 254,
    158, 193, 242,   8,  68,  27,  97, 198, 111, 210,  99,  19, 241,  76, 180,  21,  90, 239, 188, 153, 254,  36, 161,  23,  71, 120, 221,  77,  15, 224, 138, 195,  43, 242,   6, 122, 110,  75,   8, 128, 183, 238,  60,  23, 197,  68, 170,  36, 143, 199, 244,  53,  74, 187, 134,  19, 179,  63,  80, 188, 114,  54, 225, 136, 246, 155, 219, 185,  61, 120, 127,  51, 159, 245, 138, 225, 162,  73, 239, 112,  33, 150, 163,  62,  77, 169,  37,  81, 111,  31, 216, 153,  11, 106, 241,  96,   6,  61, 164,  15, 190, 233, 146, 176,  47, 117, 105, 214, 189,  76,  28, 126, 234, 182,  81, 227,  61, 133, 178,  25,  99, 206, 109,  75, 171, 149, 212,  37,
    225,  46,  78, 119, 203, 155, 178, 254,   7, 121,  45,  62, 202, 130,  50, 215,  64, 169, 133, 128,  15, 136, 215, 145, 245,   9, 111, 203,  54, 178, 131, 230, 148, 185,  80, 220, 195, 167, 242, 154, 213,  14,  99, 229, 121,  94, 238, 158,  13, 132, 174,  34, 211, 159, 145, 226, 163, 241,  38, 218,  22,  84, 162,  41, 180,  31, 104,  16, 168, 226,  97, 237,  18, 175, 148,   4, 132,  43, 201,  81, 228, 196,  15, 215,  25, 199,  94, 125, 229, 198,  71, 176,  88, 203,  54, 168, 193, 143, 210, 244, 117,  69,   4, 136, 204,  81, 240,  38, 145, 161, 244,  65, 112,  53,   3, 138, 195,  35, 148, 225,  49, 127,  61, 222,  31, 195, 107,  72,
    178,  15, 126, 102, 235,  43, 142,  58,  86, 127, 223, 167, 149,   7, 142, 191, 156,   1, 226, 144, 174,  74, 191,  83, 169, 195,  49,  99, 248, 145,  37,  71,  15, 105, 159,  57,  19,  98,  39, 139,  52,  78, 113,  46, 179,  26, 112,  55, 213,  64, 151, 138, 239,  12, 203,  44,  91, 200, 118, 125,  99, 193,   4, 209,  95,  72, 196, 249,  79,  41,  25, 193, 112,  65, 204,  84, 250, 168, 143,   8, 101, 117,  83, 254, 106, 117, 240,  12,  65, 168, 143, 236,  46, 114, 122,  29, 227, 134,  26, 106,  90,  41, 185, 254,  58,  21,  99, 181, 134, 208,   9, 121,  94, 212, 150, 248,  93, 217,  70, 166, 190,   4, 122,  89, 240, 140,  24,  93,
    202, 112,  86,  34, 169,  75, 218,  24, 235, 104,  30, 187, 251, 133, 226,  37, 243,  78,  39, 209,  54, 239,   5, 109,  42, 228, 153,  72, 163,   1, 217,  94, 238, 201,  34, 247, 180, 224,  72, 192, 252, 166, 209, 149, 248,  80, 219, 177,  90, 254,  23, 183,  94,  65, 119,  78,   4, 104,  30,  59, 245, 150, 234,  65, 120, 228,  46, 111, 122, 207, 173,  74, 220, 101,  36, 181,  21,  69, 224, 182,  60,  37, 123, 127,  44, 187,  52, 178, 103,  40,  17, 135, 219,   1, 248,  81, 153, 176,  74,  49, 202, 225, 163, 130, 152, 191, 227,  14,  53, 175,  82, 224,  20, 187, 166,  28,  52, 158,  15,  86, 254, 113, 216,  20, 178, 159,  57, 244,
    165,  59, 250, 211,   1, 191, 134, 159, 201,  65, 117,  90,  69,  28, 174,  87, 110, 121,  97, 183,  24, 149, 224,  66,  96,  18, 182,  32, 208,  83, 189, 168,  50, 116,  97, 125,  84, 120,  23, 150,   7,  86,  30, 193,   1, 162,  45,  17, 109, 195,  81, 223,  49, 106, 229, 127, 252, 187, 222, 168,  75,  43, 174, 111,  27, 151, 166,   5,  96, 235, 126, 119,  10,  57, 232, 154, 210, 104,  31, 159, 243, 213,  93,  68,   4, 224,  78, 213, 116, 242, 194, 151,  76, 184, 101,  37, 203,   8, 250, 160, 145,  17, 128,  37, 220, 141,  72,  91, 156, 251,  38, 127,  48, 235,  71, 102, 184, 239, 119, 203,  36,  95,  70,  49, 131, 209,   5, 144,
    221,  19, 183, 151, 129, 239,  39, 180,  18, 124, 230,  12, 215, 158, 204,  53,  21, 199,  63, 250, 132, 128, 164, 205, 146, 252, 137, 130, 238,  43, 132, 151,  18, 213, 177,   3,  52, 211, 102, 232, 172, 105, 228,  58, 143, 135, 200, 232, 124,  59, 168,   1, 154, 199,  19, 111,  56, 157,  85,  10, 141, 207,  19,  97, 222, 189, 242,  60, 184,  23,  45,  88, 254, 190, 130, 137,  51,  89, 192, 129, 147,  13, 193, 231, 170, 154,  97,  34, 124,  83,  53, 211,  24, 159,  63, 233, 111,  87, 186,  64, 232, 134, 179,  78,  12, 167,  30, 234, 142, 192, 100, 118, 205, 124, 114,  11, 213,  77, 107,  60, 177, 146, 242, 191, 137, 234,  78,  99,
    154,  93,  39, 137,  52, 143,  71,  89, 249,  48, 102, 190,  44, 128, 139, 247, 152, 222,   9, 162, 139, 194,  51,  29, 171,  58, 190,  75,  16, 139, 230,  64, 254,  90,  67, 234, 113, 186,  33,  59, 198,  40, 115,  92, 182, 241,  66, 115,  98,  41, 210, 137, 248, 180,  40, 123, 214,  24, 178, 233, 190, 134, 254,  53,  74,  13,  84, 116, 218,  71, 200, 108, 168, 144,  18, 176, 244,   3, 231, 137,  50, 176, 134,  34, 137, 198,  19, 251, 186,   6,  99, 171, 245, 134, 192,  47, 125, 219,  21,  97,  39, 195, 217,  54, 242, 200, 136,  46, 210,   1,  68,  27,  89,  57, 251, 171,  42, 150,   8, 232,  23, 207, 165,  10, 155,  33, 185,  42,
    196,  68, 231, 203, 164, 216,  14, 155, 211, 116,  80, 171, 241, 133,   5, 186,  76, 173,  89,  44, 214,  20, 241,  81, 220,   9, 212, 160, 147, 203, 179,  25, 109, 187, 152,  32, 166, 250,  93, 157, 140, 242,  74, 218,  13, 154,  36,  81,  10, 244, 150,  30,  90,  68, 100, 237,  74, 109,  49,  95,  33, 160, 129, 199, 177, 145, 205,  38, 102, 171, 240,   1,  51, 206,  34, 217,  78, 152,  59, 203,  23, 250,  71, 150, 239,  53, 109,  73, 160, 216, 114,  70,  41, 143, 215,  14,  73, 118, 169, 240, 142,   3, 131, 150,  85, 100, 182,  66, 130, 165, 240, 183, 226, 108,  32, 201,  94, 225, 193, 157, 101,  82,  41, 129, 226,  64, 107, 238,
    135,   7, 107,  77,  25, 242, 176, 108, 193,   3, 227,  56,  22, 150, 218,  59,  31, 106, 232, 147,  67, 178, 112, 126, 102, 120,  37, 245,  60,  89,  45, 163,  79,  12, 225, 201, 106,   9,  66, 223,  16, 180,  34, 120,  52, 202, 172, 225, 193, 134, 177,  62, 231, 166, 205,   7, 185, 125, 243, 117,  63, 220,   2, 138,  32, 132, 162, 251,  15, 142, 155,  84, 228,  67, 158, 132,  40, 187,  98, 164,  82, 211, 100,  11, 214,  89, 168, 230,  59,  31, 238, 190,  10, 130, 172, 251, 102,  30, 200,  47, 157,  72, 171, 230,  42,   7, 247, 158,  21, 219, 138, 153,  15, 175,  84, 158, 115,  74,  54, 140, 249,  62, 219, 142, 194,  88,  16, 170,
    144, 254,  33, 192,  90, 113,  45,  62,  98,  37, 164, 143, 202,  82, 166, 236, 118, 206,  38, 187, 245,   5,  88,  43, 233,  69, 113, 170,   6, 227, 115, 217, 195, 119,  41,  96,  52, 148, 209,  80, 132, 161, 207,  99, 252, 106,  19, 142, 162, 128, 221,   8, 116,  47, 148,  35,  91, 200,  13, 208,  82, 149, 241, 169, 211, 234,  51,  89,  62, 222,  25, 188, 117, 103, 236, 172, 141, 249,  11, 228, 143,  43, 156, 175,  46, 117,   1, 202, 120, 103, 157,  86, 231, 152,  54,  82, 187, 226,  63,  87, 207, 254,  27, 191, 141, 213,  77, 145, 197,  38,  92,  73, 216,  45, 232,   3, 243,  26, 167,  38, 180,   1, 134, 169,  25, 247, 115, 211,
    159,  53, 178, 222, 121,  11, 199, 235,  77, 215, 136, 251,  65, 181,  26, 124,  93,  14, 162,  72,  98, 122, 227, 196,  17, 186,  93, 207, 104, 183,  34, 100,  59, 236, 125,  73, 244, 164, 182,  25, 247, 146,   6,  65, 125,  86, 235,  44,  70,  23, 199,  84, 102, 212, 176, 251, 119,  55, 169, 104,  22, 186,  39,  60,  80,  19, 188, 152, 175, 106,  75, 215,  32,  90,  14,  59, 199,  32,  75, 132, 183,  25, 232, 196,  75, 246, 182,  44,  80,  17, 177,  48, 198,  23, 210, 160,   1, 124, 115, 181,  13, 109,  91,  65, 162, 132,  32, 227,  63, 134, 254,  53, 193, 130, 145,  65, 179, 105, 210, 228,  91, 199, 154, 233,  48, 100,  75,  37,
    188,  87, 111, 126,  68, 249, 166, 139,  23, 178, 129,  10,  94,  45, 208, 111,  54, 254, 145, 222,  31, 127,  59, 169, 139, 254,  41,  21, 238,  80, 155, 250,   1, 172, 113,  14, 217, 133,  40, 129,  52, 192, 232, 114,  30,  57, 180, 212, 133, 249, 146,  38, 241,  17,  64, 106,  81, 229,  33, 247, 162, 217,  94, 113, 123,  99, 227,   5, 203, 244,  48, 169, 149, 252, 181, 109,  87, 220, 168,  57, 206,  94,  65, 106, 123,  30,  98, 226, 152, 212, 254, 142,  67, 131, 138, 234,  45, 108, 245,  34, 221, 125,  48, 226,  15, 195, 128, 180,  12, 172, 150,   6, 135, 160, 209,  38, 197,  88,  12, 120, 108,  30,  83,  64, 183, 206,   9, 225,
     70, 241,   2,  49, 104,  29, 150, 131, 228,  46, 155, 194, 222, 118, 241,   1,  75, 172, 195,  50, 116, 106,  23, 217,  76, 153,  64, 134, 144,  54,  24, 189,  91, 212,  82, 186,  28, 143, 204, 227, 168,  90,  73, 218, 162, 199, 129,   1, 159,  59, 187, 162, 137, 194, 153, 218,   1, 182, 124,  70,  48, 120,  11, 201, 246,  42, 111, 126,  36,  93, 122,   7, 196,  71,  44, 210,   5, 152, 139, 240,  16, 155, 254,   8, 127, 219,  62, 166, 137,  35,  97,   4, 222, 184,  30,  76, 193,  92,  56, 173,  95, 117, 200, 168, 147, 250,  46, 140, 237,  84, 207, 232, 177,  25, 247,  80, 116, 161, 252,  76,  52, 238, 213,  12, 131, 138, 156, 102,
     23, 207, 163, 232,  86, 189, 219,   5, 196, 146, 239,  72,  25, 165, 100, 184, 224, 136,  84,   9, 205, 249,  90, 147,   1, 209, 181, 161, 193, 216, 167, 141,  42, 107,  35, 241, 160,  87,  62,   2, 141,  22, 105, 178,  12, 247, 136, 145, 227,  26,  81, 221,  49,  75,  26, 168,  58, 208, 116,  97, 193, 238,  77, 154, 179,  64, 214,  72, 118, 188, 234,  61, 114, 226, 163, 117, 246,  51, 186,  36, 136, 176,  45, 213,  83, 115,  20, 244, 190,  57, 204, 167,  84, 150, 248, 165, 220,  19, 157, 198,  76,   7, 244,  39, 100,  67, 214, 158,  69,  26, 129,  42,  64,  93, 111,   9, 222,  49,  29, 206, 184, 165, 128, 147, 251, 173, 232,  44,
    136, 129, 142, 201,  38, 170,  53, 101,  84,  57,  13, 105, 188,  84,  58,  34, 155,  23, 238, 164,  62, 154,  48, 190, 242,  99,  49, 228,   9, 107,  68, 233, 196,  70, 221, 135,  50, 197, 254, 153, 188, 243,  66,  40, 148,  80,  47, 174, 200, 112,  99,   9, 173, 238,  96, 227, 113,  89,  16, 221,  37, 107,  55, 222,  16, 165,  28, 254,  84,  17, 104, 219,  88,  15, 102,  33, 193,  78, 107,  88, 221,  73, 145, 185,  34, 200,  91, 149,  10,  78, 117, 241,  60,  36, 129,  11,  67, 138, 236,  26, 227,  60, 106, 183,  81,   1, 189,  37, 133, 198, 140, 157, 189, 228, 123, 186,  65, 169, 147, 130,  15, 136,  36, 197,  25,  56,  83, 194,
    246, 180,  62,  14, 152, 243,  75, 121, 254, 171, 206, 115,  40, 250, 199, 142, 207,  68,  98, 138, 181, 231,  32, 171,  67,  27, 114,  79,  36, 246,  93,  16, 152, 173,   7, 147, 177,  16, 110,  78,  32, 208, 137, 234, 193, 215,  16, 237,  66,  36, 254, 151, 203, 108, 122,  34, 196,  44, 254, 156, 172,   4, 187, 137,  85, 102, 194, 151, 207,  52, 164,  27, 183, 152, 239,  67, 170,  20, 234, 201,   6, 162, 239,  61, 109, 228, 174,  42, 233, 123, 106,  18, 181, 215, 134, 207, 148, 183, 128, 134, 154, 142, 215,  29, 234, 139, 129, 243, 168, 225,  11, 246,  83,  48,  19,  99, 244, 200, 138, 226,  61, 244, 157,  73, 223, 107,   5, 153,
     53,  25, 225, 128, 133, 206,  18, 110,  32, 218,  95, 161, 226, 139,   8, 130, 244,  39, 190, 217,  19,  79, 107, 221, 121, 127, 237, 201, 125, 118,  45, 214, 131, 250,  56, 129, 236,  71, 217,  99, 171,  53, 130, 160,  23, 109,  95, 127, 117,  86, 190, 140,  57,  15,  81, 244,  69, 177, 142,  63,  84, 210, 148, 244,  39, 231, 141,   2, 174, 237, 146, 210,  74,  39, 201,  93, 223, 140, 154,  41, 102, 192,  25,  95, 123,   3,  74, 157, 196,  89,  50, 225, 141, 156,  55, 173,  39, 254,  54, 209, 187,  15, 167,  88, 158, 211, 148,  24,  57,  76, 149,  35, 205, 172, 218, 126,  32,  78,   6, 179,  83, 210,  46, 180,  90, 145, 170, 216,
    102,  80, 170, 196, 141,  45, 230, 186, 125,  70,  49,  18,  76, 154,  64, 177, 162,  87,   6, 150,  52, 118, 201,  21,  83, 103,   6, 178,  96,  63, 185, 144,  30,  81, 200, 159,  26, 185,  41, 156, 239,   7, 184, 143,  50,  74, 122, 221,  54,   6, 159, 229,  92, 213, 183, 157,   5, 129, 216,  19, 235, 109,  27, 167,  75, 178,  56,  99,  70, 133,  41, 128, 249, 136, 161,   3,  52, 132, 180,  72, 251,  51, 120, 232, 210,  54, 252, 103, 217,  24, 168, 190,  72,   8, 243,  82, 100,   1,  86,  33,  69, 247,  45, 194,  61,  11, 178,  93, 218, 104, 184, 136,  66, 107,  86, 119,  53, 159, 238,  98,  24, 150, 131,  11, 247, 205,  70,  36,
    126, 112, 252,   5, 159,  69, 168,  83,   1, 196, 237, 172, 192,  31, 235, 211,  47, 107, 230, 172, 244,  97, 125, 254,  46, 209, 166,  50, 221,  13, 205, 165, 228, 134,  40, 139, 224,  88, 115, 201,  64,  91, 228, 209, 169, 250, 189,  31, 202, 244, 177,  22,  47, 114,  32, 222, 146, 135, 194,  42, 100, 200,  54, 114, 216,  23, 205, 243,  32, 200, 141, 178,  21,  56, 219, 145, 196, 241,  13, 214, 112, 127,  88,  14, 160, 177, 115,  17,  65, 150, 248,  34,  97, 228,  29, 194, 217, 160, 230, 174, 108,  93, 119, 237, 105,  77, 254,  41, 195,   5, 233, 165,  28, 254,   1, 232, 189, 212, 109,  44, 201, 140, 228, 162,  59,  28, 188, 231,
     13, 210,  59, 101, 236,  26, 219, 103, 247, 115, 123, 101, 221, 113,  93,  15, 117,  76, 199, 141,  34,  64,   3, 183, 160,  69, 240,  87, 153, 252, 101,  73,   3, 176, 210,  64, 169,   5, 249,  30, 110, 124,  21,  41,  97,   1,  62,  89, 105, 124,  74, 101, 196, 246,  66, 170,  49, 231,  77, 156, 174,  68, 249,   9,  94, 156, 110,  82, 165, 226,   6, 155, 231,  94, 171,  80,  32,  61, 167,  93,  26, 207,  66, 197,  36,  80, 202,  42, 186, 136,  79, 208, 124, 117, 167, 106,  63, 143, 133, 198,  10, 217, 126,  17,  37, 202, 134, 164, 151,  86,  51, 211, 145, 195,  45, 114,  63,  17, 171,  69, 252, 176,  31, 195, 109, 123,  95, 116,
     67, 189,  39,  85, 182, 111, 150,  36,  63,  90,  41,  10,  81,  58, 198, 148, 254,  31,  57, 160, 224, 112,  86, 227, 145,  16, 138,  35, 194,  24, 121,  54, 243, 151,  24, 238, 100, 148,  57, 179, 232,  72, 199, 118, 235, 150, 179, 229,  16, 115,  36, 224, 125, 119, 103,  10,  90, 180,  17, 242,  29, 147, 136, 185, 226,  47, 192,  12, 117,  51,  89, 188,  64, 207,  14, 255, 128, 154, 227,  43, 121, 174, 245, 103, 122, 231,  99, 240, 161, 222,  13, 107,  43,  74, 185,  48,  18, 238,  38, 155,  76,  56, 112, 225, 167, 144, 229,  18,  59, 245, 114,  21,  74, 157, 177,  96, 127,  87, 227, 116,   3,  83,  52,  97, 236,   6,  76, 243,
     30, 166, 146, 215,  17,  52, 207, 191, 171, 225, 204, 157, 183, 241,  40, 162, 180, 217,  95,  12, 182, 208, 121,  43, 192, 129, 219, 174,  74, 111, 127,  90, 193, 130,  49,  82, 196, 218,  85, 153,  13, 167, 104,  84,  56, 214, 132, 165,  50, 240, 183,  61,  18,  80, 190, 211, 252, 110,  61,  97, 219, 196,  81,  35, 150,  69, 252, 172, 209, 107, 247,  27, 122, 101,  53, 186, 133, 141, 199,  75, 235,   0,  52, 164,  17,  69, 119,   4,  85, 144,  63, 237, 198,   4, 251, 209, 115, 178, 204, 129, 251,  23, 190,  82,  49, 185,  65, 105, 216, 175,  96, 123, 237,  55, 219,  31, 247, 201,  35, 125, 197, 111, 122, 211,  63, 181, 215, 155,
    198,  91, 241, 137, 162, 249,  75,   7, 136,  26,  55, 232, 143,  20, 137,  82,   4, 111, 121, 241,  80,  26, 104,  68, 248,  32, 157,  56, 233, 206,  40,  14, 216, 137, 184, 161,  12, 108,  36, 192, 220,  44, 255,  31, 191,  23,  70, 145, 194,  84, 153, 216, 172, 239,  26, 157,  35, 122, 191, 116,  50,   0, 164, 237, 138,  20, 101,  79,  25, 124,  72, 201, 114, 234, 150,  25, 223, 173,  10,  98, 183, 109,  83, 226, 181, 212,  54, 168, 200,  48, 183, 149, 164,  95, 123,  83, 103,  73,   6, 142, 166, 211,  99, 155, 235,   0, 119,  83, 198,  48,   8, 186,  85, 199,  14, 109, 122,  79, 102,  58, 223,  20, 245,  33, 161, 140,  20, 133,
    229,  45,   9,  72, 186, 106,  92, 230, 146, 160,  94, 108,  71, 169, 204, 225,  67, 192,  59, 156,  49, 166, 234,  12, 148, 133, 199,   0, 104, 125, 248,  68, 147, 230,  21, 255,  54, 144, 243,  67,  91, 120, 210, 176, 155, 141, 247,   9, 211,  31, 107,   2,  94,  52, 142,  68,  87, 205,  12, 227, 126, 105, 209,  62, 175, 215, 156, 225,  54, 236, 175,  42,  86,   0, 165, 205,  72,  33, 247,  58, 118, 213,  30, 140, 151,  25, 243, 108,  31, 255,  98,  20, 231,  57,  26, 220,  36, 243,  54, 221, 133,  43,  68,  29, 201,  94, 249,  23, 144, 162, 241, 108,  35, 118, 126, 229,  62,   8, 235, 185, 160,  86, 179,  77, 201,  43, 255,  79,
    141, 178, 131, 220,  55,  28, 123,  46, 178, 251, 195,   3,  38, 245, 132,  43, 102, 248,  25, 222, 202,  92, 184, 213,  76, 178, 237,  90, 118,  50,  97, 177, 163,  37,  78,  99, 205, 179, 165,   9, 125, 109,  60,   5, 129, 221,  48,  95, 171,  65, 255, 116, 207, 164, 222, 179, 232,  44, 168,  91,  66, 251,  26,  94,  46, 197,   4, 116, 185,  97,  14, 161, 225,  65, 245,  48,  92, 126, 104, 201,  21,  68, 255, 133, 205,  77,  94, 178, 214,  73, 114,  44, 203, 111, 179, 156, 192, 147, 173,  88,  14, 181, 245, 143, 171, 112,  43, 182, 223,  32,  64, 205, 226,  73,  96,  43, 192, 171, 150,  29,  47, 217, 106,  10, 150, 128, 170,  54,
    193, 151,  24, 160, 196, 240, 114, 215,  16,  62,  84, 223, 182, 128, 150,  16, 177,  90, 114, 123,  16, 112,  40, 140, 160,  45,  21,  69, 224,  26, 212,   7, 200, 106, 119, 126,  64,  32, 215, 100, 232,  28,  80, 240, 135, 198,  76, 111, 228, 156,  45, 183,  75,  38, 134,   6, 101, 118, 243,  35, 178, 119, 191, 113, 125, 241,  90,  68,  34, 204, 112, 213, 147, 182, 104, 117, 194, 121,  40, 229, 142, 190, 167,  36,  59, 192,  17, 117, 124,  10, 226, 176,  90,  67, 248,   9,  60, 136,  30, 235, 200, 138, 129, 219,  12,  70, 207, 156, 136,  82, 100, 150, 173,   0, 250, 156, 210, 129, 138, 251,  97, 125,  64, 241, 223, 136, 209,   0,
    226,  67, 250, 139,  83,   9,  68, 100, 200, 126, 111,  51, 160, 141, 200, 230, 159, 213,  52, 182,  79, 252,  66,   7, 241, 108, 206, 171, 144, 188, 156,  87, 244,  48, 218,   0, 234, 122,  81,  50, 181, 161, 201, 144, 165,  39,  19, 122, 185,  13, 101, 234,  24, 152, 248, 191,  53,  79, 125, 211,  16,  76, 230,   9,  79,  32, 121, 169, 248, 123,  81,  55,  95,  19,  41, 216,   8, 242,  83, 173, 155,   5,  85, 236, 130, 164, 233,  41, 102, 194, 157, 142,  14, 212, 120,  96, 230, 209,  79, 154,  61,  25, 158,  46,  86, 236, 103,  58,   9, 255, 193,  24,  59, 138, 181,  30,  70,  15, 221,  76,   0, 206, 117,  38, 176,  28,  70, 109,
     31,  99, 205,  45, 180, 145, 223, 169,  29, 119, 242,  23, 212,  10,  55,  85,  29,  70,   0, 228, 100, 195, 149, 223, 186,  97,  53, 255, 130,  56, 236,  35, 124,  70, 169, 184,  87, 110,  21, 249, 149, 221,  52,  14, 185, 225,  98, 246,  67, 217, 149,  85, 197, 130,  63, 141, 217,  22, 111, 189, 102,  52, 161, 204, 176, 105, 220, 194,  48,   7, 230,  31, 255, 196, 168,  80, 159,  62,  22, 207,  48, 105, 220, 143,  11, 215,  55,  84, 247,  65,  32, 240,  52, 162,  76,  34, 114, 169,   0, 188, 110, 239, 205, 175, 147, 187,  21, 229, 179, 143,  41, 235, 158, 212,  83, 132, 241, 144,  50, 197, 172, 109,  54, 193,  91, 158, 246,  85,
    183,  56, 113,  17, 231, 158,  42,  91, 210,  58,  93, 190,  80, 255, 118, 107, 239, 192, 136, 164,  45,  25, 169, 131,  33,  81,  14, 134, 165,  21,  74, 117, 193, 100,  25, 246,  55, 206, 189, 133,   4,  73, 101, 251,  83,  60, 116, 126,  35, 200,  49, 169,   8, 213, 157,  16, 170, 237,  60,  87, 247, 216, 146,  44, 250,  60,  13,  94, 127, 102, 188, 154, 138,  72, 237, 108, 222, 184,  98, 114, 247,  76,  53, 178, 158, 132, 140, 184,   3, 218, 110, 189, 103, 229, 180, 201,  48, 102, 252,  70,  37,  97,   9,  63, 251,  41, 115, 167,  94, 216,  73, 107, 187,  15, 233,  54, 189, 166,  90, 154,  23, 234, 214,  12, 146, 218,  16, 168,
    210, 238, 127,  94, 198,  76, 255,   2, 177, 234,  37, 105, 169,  63, 125,  44, 173, 149, 128, 246,  72, 210, 138,  58, 204, 155, 219, 146, 197, 228,  96, 212,  10, 230, 158, 119,  96,  40, 156, 141, 211,  36, 124, 114,  23, 160, 209,   0,  92, 109,  75, 250, 135,  41, 235,  85, 193, 150,  35, 164,   0, 139,  28, 128, 153, 183,  77, 242, 115,  61, 217, 170, 130,   4,  56,  26, 125,  42, 234, 123,  17, 191, 209,  33, 250,  22, 206, 152, 172,  76, 148,  40,  87,  26, 149,  12, 236,  85, 158, 210, 142, 225,  81, 119, 102, 207,  84,  30,  62, 159,   6, 116,  94,  46, 151, 139,  21, 218,  36, 247,  65, 101,  83, 165,  68,  46, 128, 143,
     42,   5, 121,  61,  29, 115, 104,  66, 140, 133, 162,   8, 227, 201,  91,  13, 220,  32, 202, 142,  12,  87, 241,   4, 175, 237,  39, 183,   5,  47, 127, 111, 177,  43,  63, 213,   8, 226, 175, 242,  86, 108, 198, 173, 234, 190,  48, 240, 177, 229,  22, 183, 128, 164,  69,  32, 133, 128, 222, 182,  73, 198, 132, 234,  18, 137, 213,  28, 204,  87,  16,  42, 225, 142, 206,  92, 117, 202,  86,  36, 170,  96, 120, 111,  66, 103,  51, 242,  97,  20, 210, 166, 255,  65, 214, 129,  59, 189,  21,  55, 176, 152, 193,  49, 162,   0, 225, 190, 244,  47, 208, 240,  77, 224, 172, 203,  98,  76, 135, 191, 147,  30, 184, 252, 132, 231, 195, 135,
    219,  81, 103, 245, 176, 212,  20, 195, 151, 184, 217,  54, 112,  25, 185, 242, 103,  88,  53, 177, 226, 102, 188, 144,  70, 103, 115,  65, 243, 120,  81,  28, 248,  94, 197, 109, 127,  76,  18,  59,  44, 230,  10,  64,  31,  90, 112,  69, 156, 138,  62, 146, 225, 197, 139, 210, 252,   9, 141,  43, 239, 152,  63, 172, 196,  55, 165, 149,  44, 122, 240,  73, 185, 153, 174, 228,  69,   7, 180, 217,  59, 240,   0,  81, 231, 193,  88,  31, 119, 234,  55, 191,   5, 141, 133, 175, 152, 136, 228, 105,  29, 247,  17, 216, 182,  73, 150, 132, 142, 128, 165,  21, 192,  31, 131,   3,  58, 240, 170,   7, 131, 224,  49, 138,   3, 176,  29, 160,
     66, 181, 201,  41, 146, 160, 226,  56, 245,  24,  73, 125, 248,  48, 161, 148,  65, 213,  23, 155,  67,  31, 159,  44, 215,  21,  84, 125,  97, 201, 226,  62, 147, 161,  73,  27, 255, 123,  98, 188, 142, 155, 209, 102, 122, 221, 125,  14, 204,  44, 214,   6,  82,  46,  16, 148,  56, 167, 205,  88, 103,  16, 210,  37,  94, 223,  12, 237, 177, 195, 114,  98,  27, 245,  16,  46, 127, 252, 106,  75, 158, 139,  45, 214, 165,  14, 156, 220,  68, 107, 124,  94, 158, 198, 224,  31, 249,   4, 202,  81, 121,  65,  93, 111,  38, 246,  25, 170,  12, 199, 138, 130,  55, 140, 252, 156, 209, 142,  50, 231, 160, 205,  74, 151, 209,  62,  90, 247,
    152,  24, 141, 228,  14,  71,  91,  37, 109,  94, 201, 117, 101,  80, 208, 138,   7, 109, 247, 194, 117, 208, 252,  95, 170, 240, 192,  30,  50, 108,  15, 186, 215,   7, 233, 171,  51, 113, 206, 239, 167,  22,  80, 248, 180,  73,  36, 255,  96, 186, 160, 246, 100, 174, 237,  94, 187,  72,  27, 229,  51, 161, 255, 113,  72, 123, 106,  66,  92,   0,  58, 202, 125,  78, 115,  99, 195,  33, 119,  14, 226, 188, 149, 179, 132, 141,  46, 177, 197,   9,  42, 228,  63,  21,  82, 106,  53, 179, 162,  40, 237, 200, 172, 232, 125,  90, 209,  63, 230,  40, 249, 178, 150, 216,  68, 182,  82,  20, 195,  70,  91,  14, 174,  41, 239, 106, 116,  10,
    224,  96, 170,  55, 100, 189, 117, 173, 231,  17,  64, 223,   0, 178,  28, 233, 182, 163,  75,  99,  48,  15, 111,  81,  11, 133, 157, 209, 255, 151, 168,  91,  41, 107, 190,  87, 218,  36,  90,  65,  39, 194, 119,  49,   4, 197, 108, 170,  78,  19, 106,  30, 193,  60, 156,  35, 226, 130, 155, 179, 144, 191,  86,   6, 127, 244,  32, 118, 211, 153, 255,  41, 108, 216,  52, 236, 123,  80, 174, 202,  54,  21, 247,  32,  61, 201, 252, 130, 149, 241,  82, 115, 183, 245,  41, 121,  89, 214, 145,  98,   9, 113,  52,  13,  71, 106, 118, 183, 104,  92,  69,   6, 201,  34, 101,  45, 232, 130, 153,  27, 105, 249, 133, 195,  81,  33, 212, 184,
     82,  38, 238, 120, 210, 250,   4, 203, 148, 163, 187,  42, 158, 239,  97,  40,  59, 217,  18, 123, 235, 167,  60, 200, 229,  52, 141,   3, 177,  66,  31, 236, 123,  71, 117,  19, 148, 182,  10, 118, 221, 107, 236,  96, 158, 228,  55, 217, 141, 237,  66, 119, 220, 112,  80, 203,  15, 138, 247,   4,  68, 220,  44, 176, 202,  49, 171, 229,  23,  69, 169, 225,  17, 190, 171,   3, 212,  60, 244,  94, 114, 123,  75,  92, 222,   6,  79,  24, 134, 172, 208,  16,  99, 169, 205, 127, 237,  24,  70, 224, 191,  84, 162, 221, 198,  28,  48, 216,  17, 115, 219, 156,  89, 243, 116,  11, 162, 137, 243, 210, 179,  35, 142, 158,  17, 172, 148,  47,
    118, 204,  11,  74, 110,  46,  86,  63,  39, 135, 255, 143, 204,  70, 125, 117, 198, 149,  36,  91, 218, 183, 143,  33, 152, 179, 219,  77,  96, 224, 114, 203,  49, 250, 210,  59, 230, 162, 242, 125,  72,  16,  58, 175,  23,  83, 146,   8,  45, 155, 211, 171,  50,   0, 251, 176, 147,  55, 209,  99, 120,  28, 104, 231, 116,  98,  77, 194, 104, 184,  85, 124,  97,  65, 154,  89, 117,  26, 159,  37, 230, 104, 209, 170, 112,  99, 163, 232,  62,  31, 157,  52, 220,  68, 117,   9,  57, 188, 158,  45, 120, 252,  32, 139, 150, 176, 248, 158,  58, 238,  42, 172,  25,  74, 125, 219, 198, 176,  47, 114,  60,  87, 225, 206,  97, 229,  62, 255,
    105, 124, 187, 164,  24, 181, 143, 237, 210,  76,  10,  87,  51,  14, 108,  77, 227, 173, 251, 114,  71,   9, 131, 248,  70,  27, 109, 121,  44, 191,  13, 100, 179,   3, 164,  99,  30, 202,  51,  92, 180, 208, 144, 196, 135, 244, 178, 203,  98, 188,  34,  92, 235, 123, 105,  42,  88, 193,  34,  75, 237, 199, 125,  57,  21, 214,   8, 151,  38, 236,  11, 113,  31, 247, 218,  46, 104, 194, 221,  82, 181,   4,  65,  25, 242, 191,  40, 203, 139, 185, 246, 109, 124,  26,  93, 227, 105, 143, 245,  18, 107,  60, 184,  76, 234,  10, 120,  82, 200, 181, 142, 132, 190, 234,  53, 108,  35,  76,   8, 101, 233, 186,   0,  47,  70, 137, 194,  22,
     50, 218,  58, 246,  95, 224, 155, 133,  23, 182, 106, 165, 231, 188, 248,  26,  94,   3,  57, 126,  41, 197, 159, 213,  98, 190, 229,  18, 247, 165,  70, 154,  81, 141, 193,  79, 116, 123, 106,   4, 251, 160,  31, 233,  75,  34, 131,  62, 250,  75,  10, 114, 190,  69,  21, 219, 160, 243, 113, 181,  17, 112,  77, 245,  90, 179, 141, 250,  56, 159, 211,  59, 196, 141, 178,  19, 239, 151,  11, 139,  47, 255, 198, 155,  50, 144,  87,  12, 221,  73,  94,   0, 188, 240, 173,  33,  78, 170, 200,  86, 216, 167,   2, 210,  96,  66, 110,  36,  97,   0,  73, 213, 147,   8,  93, 122, 255,  95, 218, 151,  26,  74, 154, 131, 244,  11, 151,  91,
    168,   0,  85, 114,  40, 200,   7, 174,  92, 226,  32, 215, 150, 137, 170,  50, 157, 192, 104, 211,  85, 236, 139,  45,   6,  84,  55, 127, 102, 209,  26, 240, 217,  38, 231,  21, 247,  67,  39, 223,  78, 111,  99,  51, 169, 215, 153,  16, 166, 143, 218, 162,  38, 207, 151, 184,  60,   9,  83, 163, 221,  50, 189, 160,  35, 222, 133,  73, 202,  94, 118, 230,  79, 163, 133,  61,  85, 186,  67, 210, 167, 147, 129, 136, 183, 230,  59, 171, 153,  34, 118, 228,  84,  43, 199, 155, 213,   5,  61, 122,  38, 238, 103, 126,  49, 189, 208, 230, 163, 255, 105,  33, 228, 158, 206,  27,  66, 184, 139, 199, 170, 250, 140, 189, 163, 216, 179, 241,
    110, 227, 152, 192,  66, 122,  77, 249,  48, 139, 190,  57, 102,  38, 212, 131, 223,  65, 243,  31, 116,  17, 187, 129, 154, 240, 205, 122,  63,  87, 181,  51, 131, 171,  61, 107, 178, 215, 148, 166, 195,  18, 124, 221,   2, 140,  53, 208, 128, 233,  51,  85, 245, 101,  78, 229, 111, 121, 233,  38, 149,  94,   7, 208, 147,  60, 164,  28, 176,   0,  49, 181,  17,  40, 205, 227, 142,  39, 246, 132,  89,  19, 217,  43,  11, 108, 206,  97, 255,  57, 196, 166,  65, 123, 112, 255,  47,  98, 235, 110, 187,  69, 120,  22, 250, 166,  16, 139,  64,  23, 173,  54,  89,  68, 179, 236, 165,  19, 230,  55,  97,  39, 207,  24,  54,  87,  30,  67,
     40, 183,  29, 241,  13, 214, 106,  30, 160,  69, 243,   0,  89,  72,  18, 177, 139,  13, 148, 182,  99,  67, 166, 227, 142, 174,  36, 113,   0, 227, 145, 135,   8, 152, 209,  90,  11, 137, 187,  25,  54, 235,  64,  89, 191, 246,  81, 182, 136,  24, 196, 111,  26, 169,   6,  49,  95,  30, 192,  62, 172, 255, 107,  71, 240,  14, 196, 226, 110, 240, 157, 103, 255,  95, 152,   4, 131, 158,  15, 179,  55, 237,  77, 171, 243,  70,  24, 142, 182,   8, 104, 212,  15, 100,  27,  67, 126, 190,  26, 172,  13, 202,  91, 220, 111,  86,  43, 156, 223, 192, 135, 241, 197,  18, 107, 117,  48, 153,  84,   7, 119,  66, 110, 226, 104, 124, 115, 195,
     77, 102, 142, 170,  97, 117, 231, 177, 208, 146, 131, 172, 207, 240, 153, 232,  79, 204, 165,  54, 215, 251,  26, 200,  49,  17,  79, 255, 199,  46, 159, 244, 200,  75,  34, 238, 161,  45, 245,  95, 120, 208, 177, 113,  36, 159,  21, 227,  45, 154, 175,  64, 212, 140, 194, 251, 174, 212, 146,  88, 204,  24, 121, 180,  43, 102, 123,  86,  65,  37, 213,  71, 116, 219,  56, 170, 235, 200,  79, 224, 135,  27, 201, 145,  90, 214, 168,  39,  78, 230, 124,  46, 239, 189, 220,  85, 116, 227, 124,  80, 242,  46, 163,  33,  64, 183, 239,  98,  77,  39, 145, 129, 162,  42, 221,  96, 196, 248, 109, 216, 187, 240,  16, 173,  73,   7, 233, 211,
     12, 237,  50, 208,  76,  44,  17,  84,  53,  24, 224, 155,  35, 128, 189,  46, 101,  25, 234,  91,   4, 127, 104,  83, 115, 219, 184, 163,  98,  76, 187,  21,  93, 225, 189, 140,  62, 206,  78, 111, 127,  11, 255, 123,  60, 205, 105,  93,  70, 255,   0,  93, 237, 154, 133,  68,  18, 136, 242,   0, 113,  57, 231,  90, 205, 115, 236,  20, 127, 120, 184,  15, 194,  32,  89, 190,  25,  99,  44, 144, 160, 185, 101,   0,  52, 135, 151, 238, 200,  95, 115,  72, 172, 139, 158,   8, 203,  35,  62, 115, 101, 213, 146, 194, 231,   5, 146, 206,  14, 217, 179,   8, 229, 148,  77,   3,  63,  38, 175,  30, 147, 163,  81, 199, 255, 150,  44, 157,
    136,  86, 159,  22, 255, 187, 150, 199, 246, 102,  64, 195, 140, 133,   6,  64, 120, 127,  42, 189, 112, 122,  35,  56, 244,  92,  62, 108,  13, 221,  37, 113, 127,  51,  17, 173, 128,   0, 228,  36,  67,  90, 106,  27,  79, 238, 174,  11, 214, 138, 203,  41, 105,  16, 219,  85, 157,  45, 165,  76, 220, 194,  37, 167,   4,  78,  55, 190,  98, 246,  87,  51, 163, 146, 244,  71, 111,  60, 252, 206,  35,  71, 220, 153, 249, 187,  14, 107,  61,  19, 217,  37,  87,  22, 248,  54, 110, 170, 246,   0, 182,  59,  17, 135,  74, 131, 170,  51, 116, 104, 250,  83,  57, 201, 181, 243, 160, 210,  91,  68, 229,  98,  42, 139,  56, 185,  93, 175,
     26, 220, 182,  67, 144, 128, 135, 163,   7, 116,  90,  15, 243, 165, 205, 255, 108, 214,  76, 244,  63, 207, 232, 178, 160,   5, 213,  42, 143, 166, 249,  64, 122, 102, 247, 149, 133, 194, 154, 178, 221, 199,  52, 226, 188, 151,  56, 145, 187, 131, 163,  81, 186,  56, 177,  30, 201, 228, 186,  32, 124, 103,  69, 150, 248, 213, 171,  36, 206,   6, 112, 231, 203, 104,   7, 222, 143, 187, 168,   6,  93, 240,  56, 170,  37,  77, 223,  44, 176, 156, 185, 242, 153, 208, 181,  99, 225,  82, 196, 145, 220,  90, 252, 174, 154,  25, 246, 195,  88,  62, 168,  34, 139, 130,  26, 134, 128, 141,  16, 190, 114,   0, 212, 155,  28, 215,  65, 245,
    146,  54, 132, 204,   0, 215, 235,  56, 182, 227, 124,  49, 216,  28, 148,  82,  23, 179, 123,  13, 169,  85,  17, 145, 129, 198, 153, 239, 188, 133, 201,  89,  34, 208,  73, 183,  28, 236, 141,  19, 136, 158,   4, 166, 132,  18, 129, 223,  29,  47, 242,  25, 224, 151, 248, 103, 119,  54,  92, 117, 250,  13, 184, 136,  29, 132, 146, 229, 159,  73, 126,  30,  61,  81, 180,  42, 157,  21, 219, 115, 127, 121, 197,  22,  98, 207, 119,  90, 251, 142, 131,   0,  62, 133,  42,  74, 120,  29,  52, 159,  24, 111,  40, 203, 222,  56, 141,  33, 232,  18, 207, 151, 237, 216,  51, 152, 232, 170, 223,  51, 249,  75, 169, 241, 106, 121,   5, 197,
    169, 236,  31,  89, 167,  49,  81, 105,  32, 203,  75, 111, 180,  71,  52, 194, 228,  94,  38, 219, 108,  49, 189, 224, 138,  30,  58,  84,  19,  68, 153,  14, 236, 170,  10, 222,  59,  93,  46, 249,  64, 129, 242, 142, 214,  43, 250, 165,  86,  67, 143, 198, 134,  65,   4,  88, 235, 127,  10, 203,  50,  88, 225, 200, 164,  65,  10, 103,  48, 182, 219, 119, 248, 166, 213,  91, 241,  74, 101,  51,  80,  13, 105, 230, 114, 125,  65,   6, 199,  32,  77, 231, 191, 140, 235,  12, 205, 125,  93, 234, 178,  79, 126,  10, 100,  82, 187, 161, 129, 135, 188,  75,   5, 161,  82,  18, 198,  39,  81, 120, 126,  38, 197,  19,  84, 231, 100,  72,
};

namespace ShadowAAA {

// ============================================================================
// KONFIGURASI
// ============================================================================
namespace Cfg {
    // 4 -> 6. Pada jangkauan 150 m, empat cascade meninggalkan lompatan lebar
    // 4,5x dari C2 (49,8 m) ke C3 (224 m) — cascade jauh jadi terlalu kasar di
    // sisi dekatnya. Enam cascade meratakan tangga itu.
    // Biaya: 2 render pass tambahan per frame, dan VRAM cascade array naik dari
    // 67 MB ke 100 MB (2048^2 x 6 x D32).
    constexpr uint32_t NUM_CASCADES   = 6;
    // Dipertahankan 2048 — ini nilai untuk desktop, dan menurunkannya cuma demi
    // Termux berarti harus dinaikkan lagi nanti. Optimasi performa yang tetap
    // dipakai adalah yang TIDAK mengorbankan kualitas: penghapusan exp() dan
    // pengambilan texel ganda di pass blur.
    constexpr uint32_t CASCADE_RES    = 2048;

    // Jarak maksimum yang di-cover cascade. Memakai far plane kamera penuh
    // (100 m) membuang resolusi ke area yang bayangannya tak terlihat.
    // Jangkauan bayangan kelas OPEN WORLD.
    //
    // Ini aman untuk kualitas, dan alasannya penting: CSM menjaga rasio
    // texel-per-piksel tetap ~1 di semua cascade. Cascade terjauh memang punya
    // texel raksasa (~15 cm di 150 m), tapi satu piksel layar di jarak itu juga
    // mencakup ~17 cm dunia — jadi ketajaman yang TERLIHAT tidak berubah.
    // Memperluas jangkauan tidak menurunkan mutu per piksel; ia cuma memindahkan
    // resolusi ke tempat yang lebih jauh.
    //
    // CATATAN: nilai efektif = min(nilai ini, Config::CAMERA_FAR_PLANE).
    // Far plane kamera saat ini 100 m, jadi naikkan itu juga saat dunianya
    // membesar. Jarak efektif dicetak ke log waktu start.
    // v41: 150 -> 170, sedikit di atas CAMERA_FAR_PLANE supaya batas atas
    // jangkauan ditentukan far plane (yang punya alasan geometris), bukan
    // angka ini. Nilai efektifnya tetap min(fitDist, farZ).
    constexpr float SHADOW_MAX_DISTANCE = 170.0f;

    // ---- KECEPATAN ORBIT MATAHARI ---------------------------------------
    // Dulu sudutnya time * 0.1 rad/s, yaitu satu putaran penuh tiap 63 DETIK.
    // Itu mustahil dipakai bersama akumulasi temporal: riwayat dirata-ratakan
    // selama 40 detik, sementara mataharinya sudah berputar 229 derajat dalam
    // rentang itu. Bayangannya akan tertinggal jauh, bukan menghalus.
    //
    // Keterlambatan bayangan = jendela rata-rata x kecepatan ujung bayangan.
    // Untuk bayangan terpanjang 6,7 m dan jendela 40 detik:
    //
    //   periode orbit   gerak matahari/menit   keterlambatan bayangan
    //      63 detik          343 derajat            2673 cm   <- lama
    //       1 jam              6 derajat              47 cm
    //       3 jam              2 derajat              16 cm
    //       6 jam              1 derajat               7,8 cm  <- dipilih
    //      12 jam            0,5 derajat               3,9 cm
    //
    // Enam jam dipilih sebagai titik seimbang: keterlambatan 7,8 cm masih di
    // bawah texel cascade terjauh (2,68 cm ... 7,8 cm sebanding), dan matahari
    // tetap terlihat bergerak kalau ditunggu — 1 derajat per menit, ujung
    // bayangan bergeser sekitar 12 cm per menit.
    //
    // Naikkan angkanya kalau ingin bayangan lebih bersih lagi; turunkan kalau
    // ingin matahari lebih hidup. Pertukarannya persis tabel di atas.
    constexpr float SUN_ORBIT_PERIOD_SEC = 6.0f * 3600.0f;

    // Memaku sudut matahari supaya screenshot antar-run bisa dibandingkan.
    // Lihat catatan panjang di updateGameState(). true = matahari diam.
    constexpr bool  SUN_FREEZE           = true;
    constexpr float SUN_FREEZE_ANGLE_DEG = 45.0f;


    // 0.75 -> 0.88. Makin mendekati 1 makin logaritmik, artinya makin banyak
    // resolusi dialokasikan ke cascade dekat. Wajib dinaikkan bersamaan dengan
    // jangkauan: dengan lambda 0.75 di 150 m, cascade 0 akan membentang 0,1-10 m
    // dan bayangan kontak di kaki pemain jadi kasar.
    constexpr float CASCADE_LAMBDA      = 0.88f;
    // Ruang ekstra di belakang cascade untuk caster yang berada di luar frustum
    // kamera tapi tetap menghalangi cahaya. Dipakai yang lebih besar antara
    // nilai tetap ini dan pecahan dari lebar cascade — kalau tidak, cascade
    // terjauh (lebar ratusan meter) akan memotong caster tinggi seperti gedung
    // atau bukit, dan bayangannya hilang mendadak.
    // v48 [3]: 25 -> 60 m. Ini jarak near plane cascade DI ATAS pusatnya
    // sepanjang arah cahaya. Caster yang menembusnya dipotong rasterizer dan
    // hilang dari shadow map. Menara tertinggi peta sekarang 33 m; komponennya
    // sepanjang arah cahaya = 33 x sin(31 derajat) = 17,0 m. 25 m secara teori
    // cukup, tapi marginnya cuma 8 m dan itu tidak menyisakan ruang untuk
    // caster di luar pusat cascade. 60 m memberi margin 3,5x. Biayanya hanya
    // presisi depth, dan formatnya D32 float — jauh lebih dari cukup.
    constexpr float CASCADE_Z_PADDING      = 60.0f;
    constexpr float CASCADE_Z_PADDING_FRAC = 0.25f;

    // Bayangan meredup ke nol di N% terakhir jangkauan, bukan terpotong garis
    // lurus. Tanpa ini ada batas tajam yang sangat mencolok di dunia terbuka.
    constexpr float SHADOW_FADE_FRAC       = 0.15f;
    // Diperlebar 0.12 -> 0.20 di v6. Zona blend yang lebih lebar memberi ruang
    // lebih besar bagi sisa perbedaan antar cascade untuk menyatu perlahan.
    constexpr float CASCADE_BLEND_FRAC  = 0.20f;

    // Ukuran sudut matahari. tan(0.265°) ≈ 0.00463 = matahari asli.
    // v1 memakai 0.028 (≈6x matahari asli) supaya contact hardening kelihatan
    // jelas — kebablasan, penumbra melebar terlalu cepat dan terlalu jauh.
    // 0.012 ≈ 0.7° radius: masih sinematik, tidak lagi "blur kejauhan".
    // Tangen JARI-JARI sudut matahari. penumbraWorld = distWorld * nilai ini,
    // dan penumbraWorld dipakai sebagai RADIUS cakram filter — jadi yang benar
    // adalah setengah diameter sudut, bukan diameternya.
    //
    // Matahari dilihat dari Bumi berdiameter 31,6 menit busur = 0,5267 derajat.
    // Jari-jarinya 0,2633 derajat, tan = 0,004596.
    //
    // Nilai lama 0,009 menyiratkan diameter 1,0313 derajat — matahari 1,96x
    // lebih besar dari matahari sungguhan, dan karena penumbra berbanding LURUS
    // dengan angka ini, SETIAP tepi bayangan di layar jadi hampir dua kali
    // terlalu lebar. Itu satu-satunya sebab kenapa bayangan engine ini selalu
    // lebih lembut daripada acuan, dan kenapa mengejarnya di PCSS, blocker
    // search, maupun filter tidak pernah menemukan apa-apa: ketiganya memang
    // sudah benar, mereka cuma diberi ukuran matahari yang salah.
    //
    // Terukur pada tepi yang ditunjuk user (lebar tegak lurus, kemiringan tepi
    // sudah dikoreksi) dibandingkan acuan COD 8,3-9,9 px:
    //
    //   y=585   11,20 px  ->  5,72 px
    //   y=645   15,15 px  ->  7,74 px
    //   y=705   18,77 px  ->  9,58 px
    constexpr float SUN_ANGULAR_TAN     = 0.004596f;  // 0.012 -> 0.009 -> fisik

    constexpr float MIN_PENUMBRA_WORLD  = 0.010f;
    constexpr float MAX_PENUMBRA_WORLD  = 0.50f;  // v1: 1.10 — terlalu lebar
    constexpr float BLOCKER_SEARCH_WORLD= 0.50f;  // radius blocker search (meter)

    // Kedua bias ini SATUANNYA PIKSEL LAYAR, dikali worldPerPixel di shader —
    // bukan lagi dikali texelWorld cascade.
    //
    // Alasannya sama persis dengan MIN_PCF_SCREEN_PX di bawah: texelWorld
    // meloncat di batas cascade (11,7 mm di cascade 2 vs 31 mm di cascade 3).
    // Normal offset menggeser titik sample menjauhi permukaan, jadi kalau
    // besarnya meloncat, POSISI tepi bayangan ikut meloncat — itulah "sobekan"
    // sisa yang terukur 1,57 px dengan kedua dataran sangat bersih.
    // worldPerPixel kontinu terhadap kedalaman, jadi tidak bisa meloncat.
    // 2.0 -> 0.6. Normal offset satu-satunya bias yang MENGGESER POSISI sampel,
    // jadi ia harus sekecil mungkin. Dengan rumus slopeCorr yang baru, nilai ini
    // dikali sin/cos sudut cahaya lalu dibatasi 1 worldPerPixel — sehingga
    // permukaan yang menghadap cahaya lurus tidak digeser sama sekali, dan yang
    // menyerempet digeser paling banyak ~1 texel.
    constexpr float NORMAL_OFFSET_PX    = 0.6f;

    // 2.5 -> 4.5. Beban anti-acne dipindahkan ke sini, karena depth bias hanya
    // mengubah NILAI yang dibandingkan — ia tidak memindahkan titik sampel, jadi
    // tidak bisa mendistorsi bentuk tepi bayangan seberapa pun besarnya.
    // v48 [4]: 4,5 -> 2,0. Dengan DEPTH_BIAS_MAX_WORLD sudah 3 cm dan offset
    // caster 0,35 texel, pengali besar ini cuma membuat bias menabrak batas
    // atasnya di hampir semua cascade. 2,0 membuatnya benar-benar mengecil di
    // cascade dekat, bukan sekadar terjepit.
    constexpr float DEPTH_BIAS_PX       = 2.0f;

    // BATAS ATAS bias dalam METER — ini yang hilang sampai v13.
    //
    // Kedua bias di atas berskala worldPerPixel, yang tumbuh linier terhadap
    // jarak. Di 80 m, worldPerPixel = 9,2 cm sehingga normal offset mencapai
    // 20 cm: titik sampel digeser 20 cm dari permukaan, dan bayangan lepas dari
    // kaki caster. Itulah "bayangan jauh tidak menutup sempurna" — peter-panning
    // yang makin parah makin jauh karena tidak pernah dibatasi.
    //
    // Membatasinya aman: di kejauhan filter bayangan sudah lebar dalam satuan
    // dunia, jadi acne yang mungkin muncul kembali justru terhapus penyaringan.
    // Peter-panning tidak punya penawar seperti itu — ia terlihat apa adanya.
    constexpr float NORMAL_OFFSET_MAX_WORLD = 0.05f;   // 5 cm
    // v47: 0,10 -> 0,03 m.
    //
    // Di seluruh pipeline HANYA DUA hal yang bisa melepas bayangan dari kaki
    // casternya, dan keduanya menggeser searah cahaya:
    //   [1] depth bias penerima, dijepit oleh angka ini
    //   [2] offset sisi caster di csm_shadow.vert, sebesar CASTER_OFFSET_SCALE
    //       kali texel cascade
    //
    // Terukur pada elevasi matahari 31 derajat (proyeksi ke tanah = /tan(31)):
    //
    //   sebelum, C5 : 10,0 + 12,1 = 22,1 cm  ->  36,8 cm di tanah  (2,7 piksel di 100 m)
    //   sesudah, C5 :  3,0 +  4,2 =  7,2 cm  ->  12,1 cm di tanah  (0,9 piksel)
    //   sesudah, C0 :  3,0 +  0,1 =  3,1 cm  ->   5,2 cm di tanah
    //
    // Konsekuensinya jujur: bias yang lebih kecil = risiko acne lebih besar,
    // terutama di permukaan yang nyaris sejajar arah cahaya. Yang menutupinya
    // RPDB (yang sudah menghitung selisih kedalaman per tap secara tepat) dan
    // contact shadow di jarak dekat. Kalau muncul bintik-bintik gelap di
    // permukaan miring, naikkan angka ini bertahap — 0,05 lalu 0,07.
    constexpr float DEPTH_BIAS_MAX_WORLD    = 0.03f;

    // Konstanta WAKTU akumulasi temporal, dalam detik — bukan lagi faktor blend
    // tetap. Faktor blend tetap membuat jendela akumulasi melar saat FPS turun
    // (0.92 di 15 FPS = jendela 800 ms, sumber smear yang terlihat di lantai).
    // Faktor blend sesungguhnya dihitung di shader: blend = exp(-dt / tau).
    // Konstanta waktu dibuat sangat kecil: pass temporal bayangan jadi
    // lewat-saja. Alasannya di shadow_temporal.comp — variance clipping yang
    // tidak simetris di dua sisi tepi adalah satu-satunya beda yang tersisa
    // antara mode 3 (bersih) dan mode 0 (bertangga).
    //
    // Naikkan lagi ke 0.12 / 0.50 nanti di PC, tempat FPS-nya cukup untuk
    // akumulasi temporal benar-benar bekerja.
    // Dinyalakan atas permintaanmu. Catatan jujur supaya tidak jadi kejutan:
    // pada 0,5 FPS, blend = exp(-dt/0.50) = 0,14 — temporal cuma menyimpan 14%
    // history per frame. Itu terlalu lemah untuk menyatukan pola dither jadi
    // penumbra mulus, tapi cukup kuat untuk menghidupkan variance clipping yang
    // tidak simetris di dua sisi tepi. Di GPU sungguhan (60 FPS) blend jadi 0,97
    // dan keduanya hilang.
    // TAU dinaikkan drastis supaya akumulasi temporal benar-benar hidup pada
    // frame rate rendah. Ini yang membuat bintik penumbra hilang — bukan
    // menambah tap PCF (butuh 195 tap) dan bukan menaikkan sigmaL (jenuh 2,3x).
    //
    // Bobotnya alpha = 1 - exp(-dt/tau), dan di shader ada lantai
    // MIN_ALPHA = 0.05. Jadi ada titik jenuh yang bisa dihitung: pada dt = 2 s,
    // alpha menyentuh lantai itu tepat di tau = 39 s. Lebih besar dari itu sia-
    // sia karena MIN_ALPHA yang mengambil alih.
    //
    //   tau(s)  alpha   penekanan derau   bintik di layar   jendela rata-rata
    //     0,5   0,982       1,02x            2,43              2 s   <- lama
    //     4,0   0,393       2,02x            1,23              5 s
    //    20,0   0,095       4,47x            0,55             21 s
    //    40,0   0,050       6,24x            0,40             40 s   <- baru
    //
    // Ambang terlihat sekitar 1,0 tingkat layar. 2,43 jelas terlihat; 0,40
    // tidak. Terukur dari tangkapan layar, bukan ditaksir.
    //
    // MOVING sengaja jauh lebih kecil daripada STATIC. Saat kamera bergerak,
    // riwayat diproyeksikan ulang lewat velocity buffer, dan galat reproyeksi
    // pada 0,5 FPS besar karena kamera menempuh jarak jauh dalam dua detik.
    // 4 s sudah memberi 2x penekanan tanpa mengundang smear saat berputar.
    // Piksel yang benar-benar tersingkap tetap aman: histLen-nya kembali ke 1
    // sehingga alpha jadi 1 dan riwayatnya dibuang, bukan diblending.
    constexpr float TEMPORAL_TAU_MOVING = 4.0f;
    constexpr float TEMPORAL_TAU_STATIC = 40.0f;
    constexpr float DISOCCLUSION_REL    = 0.06f;  // ambang depth RELATIF (6%)

    // Lebar PCF MINIMUM di dalam shadow map, dinyatakan dalam PIKSEL LAYAR.
    //
    // v2 menurunkan batas ini ke 1 texel dan menyerahkan pelembutan minimum ke
    // blur layar. Itu keliru: blur layar melembutkan tepi tapi TIDAK memindahkan
    // posisinya, sehingga kuantisasi texel shadow map tetap utuh — terukur
    // sebagai tangga 1 piksel setiap ~14 piksel di tepi diagonal. Yang menghapus
    // tangga hanya PCF yang merata-rata beberapa texel.
    //
    // Satuannya piksel layar, bukan texel, supaya nilainya jadi fungsi KONTINU
    // dari kedalaman view (worldPerPixel) — bukan fungsi cascade. Itu yang
    // membuatnya tidak bisa menimbulkan pita, padahal efektifnya tetap sekitar
    // 3 texel di setiap cascade karena CSM memang menyeimbangkan texel/piksel.
    // Dinaikkan 3.5 -> 5.0 di v5. Pada 3.5 filter cuma mencakup ~2.7 texel —
    // terukur belum cukup untuk meratakan undakan siluet caster selebar 1 texel
    // (loncatan 1,11 px pada tepi yang dua sisinya datar dalam 0,1 px).
    // Bekerja berpasangan dengan TEXEL_JITTER di csm_resolve.comp: dither yang
    // memecah undakan, filter yang meratakannya.
    // 5.0 -> 3.0. Nilai 5.0 dipasang di v6 untuk menutupi undakan texel, TAPI
    // penyebab undakan sesungguhnya adalah bias yang tidak kontinu — sudah
    // diperbaiki di v7. Yang tersisa dari 5.0 cuma efek sampingnya: setiap tepi
    // bayangan dipaksa selebar minimal 5 piksel, termasuk bayangan kontak yang
    // seharusnya nyaris tajam. Referensi AAA menunjukkan tepi rapat di titik
    // sentuh lalu melebar menjauh — itu yang 5.0 hilangkan.
    // 3,0 -> 1,5 px.
    //
    // Lantai ini menjaga lebar filter tidak jatuh di bawah beberapa piksel
    // layar, supaya tepi bayangan tidak beralias di tingkat piksel. Nilainya
    // dipasang saat engine ini belum punya akumulasi temporal yang bekerja;
    // sekarang punya, dan detail sub-piksel diselesaikan lintas frame.
    //
    // Selama ini ia MENUTUPI contact hardening. Diukur pada tiang 4 m, jarak
    // pandang 4 m (worldPerPixel 0,55 cm):
    //
    //   x dari kaki   penumbra PCSS   dipakai (lantai 3 px)   dipakai (1,5 px)
    //      0,13 m         1,00 cm         1,64 cm  <- lantai      1,00 cm
    //      0,33 m         1,00 cm         1,64 cm  <- lantai      1,00 cm
    //      0,67 m         1,00 cm         1,64 cm  <- lantai      1,00 cm
    //      1,33 m         1,40 cm         1,64 cm  <- lantai      1,40 cm
    //      2,67 m         2,80 cm         2,80 cm                 2,80 cm
    //
    // Lantai 3 px menang di 57% panjang bayangan — tepatnya di bagian dekat,
    // persis tempat pengerasan kontak seharusnya terlihat. Pada 1,5 px ia tidak
    // pernah menang lagi, dan yang membatasi tinggal MIN_PENUMBRA_WORLD.
    //
    // Harganya derau. Filter menyempit dari 6,1 ke 3,1 texel di cascade 2,
    // sehingga jumlah tap turun 13 -> 7 dan derau naik 1,36x. Anggarannya ada:
    // dengan temporal 6,24x dan a-trous 2,28x, bintik di layar naik dari 0,17
    // ke 0,23 tingkat — masih 4,3x di bawah ambang terlihat 1,0.
    //
    // Perubahan ini SENGAJA menunggu sampai temporal bekerja. Dipasang lebih
    // awal, ia cuma akan menukar tepi lembut dengan tepi berbintik.
    constexpr float MIN_PCF_SCREEN_PX   = 1.5f;

    // 2.0 -> 1.5 -> 2.5. Dinaikkan lagi di v6 sebagai peredam spasial untuk sisa
    // noise dither sub-texel (terukur simpangan baku 1,0 di pita penumbra
    // melawan 0,0 di umbra rata). Tepi penumbra sekarang sudah 7-8 px, jadi
    // blur 2,5 px tidak menambah kekaburan yang terasa.
    // Radius blur minimum, DUA nilai — dipilih otomatis menurut seberapa kuat
    // akumulasi temporal sedang bekerja.
    //
    // Alasannya: kuantisasi texel pada siluet caster diredam oleh TEXEL_JITTER,
    // yang mengubah tangga menjadi noise. Tapi noise itu baru benar-benar
    // hilang kalau ada yang meredamnya — dan peredam utamanya akumulasi
    // temporal. Pada 60 FPS temporal menyimpan ~97% history tiap frame sehingga
    // blur spasial boleh sempit dan bayangan tetap tajam. Pada 0,5 FPS temporal
    // cuma menyimpan ~13%, praktis mati, dan noise jitter muncul kembali sebagai
    // tangga di tepi bayangan.
    //
    // Jadi nilainya tidak dipatok: shader menghitung kekuatan temporal dari
    // deltaTime dan menginterpolasi keduanya. Di PC nanti otomatis memakai nilai
    // tajam; di Termux otomatis memakai nilai stabil. Tidak ada yang perlu
    // disetel ulang saat pindah perangkat.
    constexpr float MIN_BLUR_RADIUS_PX      = 1.2f;   // saat temporal sehat
    // 3.0 -> 1.4.
    //
    // Pengukuran langsung mode 3 lawan mode 0 pada tepi yang sama:
    //   bayangan mentah   : tangga 0,49 px  (tertelan noise dither 1,50 px)
    //   setelah disaring  : tangga 1,61 px
    // Penyaringan MEMPERBESAR tangganya 3,3 kali.
    //
    // Sebabnya radius blur yang lebar. Blur simetris tidak menggeser tepi, TAPI
    // radiusnya di sini ikut penumbra terdilasi — dan di dua sisi tangga
    // penumbranya berbeda, jadi radiusnya berbeda, dan sisi dengan radius lebih
    // besar bergeser lebih jauh. Makin lebar radiusnya, makin besar selisih
    // pergeseran itu.
    //
    // Nilai 3.0 dipasang di v15 untuk meredam noise saat FPS rendah. Mode 3
    // membuktikan peredaman itu tidak sepadan: gambar tanpa peredaman sama
    // sekali justru terlihat lebih bersih, karena noise 1,5 px yang acak jauh
    // kurang mengganggu daripada tangga 1,6 px yang diam di tempat.
    constexpr float MIN_BLUR_RADIUS_PX_SLOW = 1.4f;
    constexpr float BLUR_MAX_RADIUS_PX  = 5.0f;   // sama dengan MAX_RADIUS shader
    // Jumlah cascade masih 4. Untuk dunia terbuka sungguhan, 6 cascade akan
    // memberi ketajaman dekat yang jauh lebih baik pada jangkauan segini —
    // itu langkah berikutnya, tapi butuh perubahan layout UBO (skalar per-cascade
    // sekarang muat di satu vec4) dan 2 render pass tambahan per frame.

    // ---- CONTACT SHADOW (ray march di ruang layar) -----------------------
    // Menangkap detail kontak yang shadow map seukuran apa pun tidak bisa
    // selesaikan: celah tipis, pertemuan objek, kaki benda kecil. Sengaja
    // DILEBUR ke dalam csm_resolve, bukan jadi pass terpisah — ia butuh depth,
    // normal, dan posisi dunia yang semuanya sudah dihitung di sana, jadi tidak
    // ada image, descriptor, maupun dispatch baru. Hasilnya juga otomatis ikut
    // diredam temporal dan blur bersama bayangan CSM.
    // AKTIF sejak v16, setelah tiga perbaikan berurutan:
    //   v10  penolakan tabrakan-diri yang sadar-bidang (ambang tumbuh bersama ray)
    //   v11  texelFetch menggantikan texture() yang menginterpolasi kedalaman
    //   v16  jitter awal ikut masuk hitungan tinggi ray, dan hasilnya diredupkan
    //        menurut jarak tempuh alih-alih biner
    //
    // Panjang ray sengaja pendek (18 cm). Contact shadow bertugas mengisi detail
    // DI BAWAH satu texel cascade — sekitar 0,3 cm di cascade 0. Memanjangkannya
    // hanya menambah peluang tabrakan-diri tanpa menambah detail, karena di atas
    // skala itu CSM sudah menanganinya.
    //
    // CONTACT_STEPS = 0 mematikannya sepenuhnya.
    // v48 [5]: 0,18 -> 0,75 m, 8 -> 14 langkah.
    //
    // Ini pengaman terakhir, dan satu-satunya jalur bayangan di engine ini yang
    // TIDAK kenal bias sama sekali: ia menelusuri depth buffer layar, bukan
    // shadow map. Apa pun yang lolos dari cascade — bias, texel, caster hilang
    // — masih bisa ditangkap di sini selama jaraknya di bawah panjang ray.
    // Inilah cara game AAA menutup meter terakhir di kaki benda.
    //
    // 0,75 m dipilih karena itu setengah lebar sel peta (1,5 m), jadi celah di
    // kaki dinding mana pun tercakup. 14 langkah memberi jarak antar-sampel
    // 5,4 cm — cukup rapat untuk menangkap dinding setebal 1,5 m dari sisi
    // mana pun, dan itu yang menentukan.
    constexpr float CONTACT_LENGTH_WORLD = 0.75f;  // panjang ray, meter
    constexpr int   CONTACT_STEPS        = 14;     // 0 = mati
    // Tebal maksimum penghalang: 0,30 -> 0,06 m.
    //
    // Nilai lama 1,7x LEBIH PANJANG daripada ray itu sendiri (0,18 m), jadi
    // praktis apa pun yang tertangkap di jalur ray masuk hitungan — termasuk
    // dinding bersebelahan di sudut bangunan. Tebal harus lebih KECIL daripada
    // panjang ray, karena tugasnya membatasi seberapa tebal benda yang masih
    // masuk akal dianggap penghalang kecil, bukan menampung seluruh jangkauan.
    constexpr float CONTACT_THICKNESS    = 0.06f;
    // v48 [6]: 25 -> 60 m. Kebocoran yang terlihat ada di bangunan JAUH; kalau
    // contact shadow sudah mati total di 25 m, pengaman terakhirnya tidak
    // pernah sampai ke tempat masalahnya.
    constexpr float CONTACT_FADE_DIST    = 60.0f;

    // ---- AMBIENT OCCLUSION (GTAO) ----------------------------------------
    // Bayangan dari cahaya LANGIT, bukan matahari. Ini celah terbesar yang
    // tersisa: ambient di engine ini masih konstan rata untuk setiap piksel,
    // sehingga bagian dalam bayangan datar tanpa struktur dan sudut pertemuan
    // dinding-lantai sama terangnya dengan tengah lantai.
    //
    // AO_SLICES = 0 MEMATIKAN fitur ini (aoOut diisi 1.0, ambient kembali rata).
    constexpr float AO_RADIUS_WORLD = 0.85f;  // jangkauan oklusi, meter
    // v68: 3 -> 6. Batas di shader sudah dinaikkan ke 8, jadi angka ini
    // sekarang benar-benar berpengaruh. Resolusi sudut 180/3 = 60 derajat
    // per irisan turun jadi 30 derajat.
    constexpr int   AO_SLICES       = 6;      // irisan bidang (maks 8 di shader)
    // v68: 8 -> 12, seiring MAX_STEPS di shader. Yang menentukan bukan
    // jumlahnya melainkan JARAK antar-sampel, dan langkah eksponensial di
    // shader membuat 12 langkah menutup celah dekat jauh lebih rapat daripada
    // 8 langkah linear.
    //
    // 6 x 12 x 2 = 144 sampel per piksel, masuk kelas HBAO+ (128-256) yang
    // dipakai RDR2. Sebelumnya 3 x 8 x 2 = 48.
    constexpr int   AO_STEPS        = 12;     // langkah per irisan (maks 12 di shader)

    // Batas atas radius AO di ruang layar. Dulu 64 px: dengan 8 langkah, celah
    // antar sampel di ujung mencapai belasan piksel, sehingga horizon di
    // antaranya tidak pernah tersampel dan tepi AO meloncat bertangga selebar
    // celah itu — persis "sobekan" yang terlihat di kaki dinding.
    // 40 -> 128. Ini pencekik, bukan penghemat.
    //
    // Jumlah sampel AO = AO_SLICES * AO_STEPS * 2 = 48 per piksel, dan angka
    // itu TIDAK bergantung sama sekali pada radiusPx — perulangannya dibatasi
    // steps, bukan jarak. Jadi memotong radius tidak menghemat satu texelFetch
    // pun; ia cuma mengecilkan jangkauan DUNIA yang diperiksa.
    //
    // Terukur pada sudut dinding-lantai di jarak 3,1 m (worldPerPixel 0,43 cm):
    //     radius yang diminta AO_RADIUS_WORLD : 199 px = 0,85 m
    //     radius yang benar-benar dipakai     :  40 px = 0,17 m  (20%)
    // Akibatnya pita oklusi cuma selebar 6 px alias 2,4 cm — di bawah radius
    // penyaring a-trous sendiri, jadi ia dihapus sebelum sempat terlihat. Itu
    // sebabnya bagian dalam bayangan terukur rata di ambFactor 0,87-0,91.
    //
    // Sapuan batas radius pada sudut yang sama:
    //     40 px -> pita 0,024 m, AO minimum 0,820, noise sd 0,0044
    //     64 px -> pita 0,087 m, AO minimum 0,740, noise sd 0,0042
    //     96 px -> pita 0,143 m, AO minimum 0,736, noise sd 0,0032
    //    128 px -> pita 0,151 m, AO minimum 0,733, noise sd 0,0031  <-- lutut
    //    200 px -> pita 0,151 m, AO minimum 0,739, noise sd 0,0030
    // Jenuh di 128; lebih dari itu cuma memperburuk lokalitas cache tanpa
    // menambah oklusi. Perhatikan noise-nya justru TURUN, karena celah antar
    // langkah yang lebih lebar membuat sampel saling bebas.
    constexpr float AO_MAX_RADIUS_PX = 128.0f;
    // 1.00 -> 2.00. Ini pencekik KEDUA, dan angkanya diturunkan, bukan dikira-kira.
    //
    // main.frag menerapkan multibounce Jimenez pada AO memakai albedo permukaan.
    // Rumus itu benar secara fisik, tapi pada albedo 0,7 (warna dinding di level
    // ini) ia mengangkat AO dengan sangat kuat dan membuang sebagian besar
    // oklusi yang baru saja dihitung susah payah:
    //     AO 0,48 -> 0,753   (tersisa 47% dari oklusi aslinya)
    //     AO 0,74 -> 0,895   (tersisa 40%)
    //     AO 0,90 -> 0,957   (tersisa 43%)
    // Jadi meskipun GTAO menghasilkan 0,48 di sudut, yang sampai ke gambar 0,75.
    //
    // AO_INTENSITY adalah pangkat yang dikenakan SEBELUM multibounce, jadi ia
    // tepat berada di tempat yang bisa membatalkan kompresi itu. Nilai 1,99
    // adalah pangkat yang meminimalkan galat antara mb(AO^p) dan AO pada
    // rentang 0,30-1,00 (galat rms sisa 0,045). Dibulatkan ke 2,00:
    //     0,48 -> 0,232 -> multibounce 0,475   (target 0,48)
    //     0,74 -> 0,549 -> multibounce 0,801   (target 0,74)
    //     0,90 -> 0,811 -> multibounce 0,923   (target 0,90)
    //     1,00 -> 1,000 -> multibounce 1,000   (target 1,00)
    // Jadi ini BUKAN menambah AO melebihi yang dihitung — ia mengembalikan
    // kontras ke persis nilai yang dihasilkan GTAO, tak lebih.
    constexpr float AO_INTENSITY    = 2.00f;  // pangkat; >1 = lebih pekat

    // Batas bawah visibilitas: AO tidak boleh menghitamkan apa pun sepenuhnya.
    // Di dalam bayangan, ambient adalah SATU-SATUNYA cahaya, jadi AO yang boleh
    // mencapai nol membuat bayangan jadi lubang hitam tanpa detail. Pengukuran
    // pada referensi AAA: bayangan menahan ~31% kecerahan area terang dan masih
    // menyimpan detail di dalamnya.
    // 0.45 -> 0.0. INI LANTAI YANG KEDUA, dan itu memang tidak disengaja.
    //
    // v18 memindahkan lantai "bayangan tetap menyimpan detail" ke main.frag
    // (AMBIENT_FLOOR), dengan alasan yang benar: lantai baru bermakna kalau
    // diterapkan pada apa yang BENAR-BENAR sampai ke gambar, sesudah semua
    // pengali. Komentarnya menulis "dipindah" — tapi yang lama tidak pernah
    // dihapus. Jadi sejak v18 ada DUA lantai berurutan, dan yang di sini
    // dipasang di tempat terburuk: sebelum multibounce Jimenez.
    //
    // Akibatnya berlipat, bukan bertambah. Multibounce menaikkan AO menurut
    // albedo, dan menaikkannya dari nilai yang SUDAH dilantai berarti membayar
    // dua kali. Terukur pada rantai ini (albedo 0,7):
    //
    //     AO mentah 0,15  ->  0,53 (lantai)  ->  0,79 (multibounce)
    //     sudut dalam cuma 19% lebih gelap daripada lantai terbuka,
    //     padahal AO-nya menyatakan 83%.
    //
    // Itulah cahaya yang bocor ke sudut dan ke dalam bayangan: bukan bayangan
    // yang salah, melainkan AO yang tidak pernah diizinkan menggelapkan apa pun.
    // AMBIENT_FLOOR di main.frag tetap menjaga bagian tergelap tidak jadi hitam
    // pekat — satu lantai, di tempat yang benar.
    constexpr float AO_MIN_VISIBILITY = 0.0f;

    // Radius blur khusus AO, terpisah dari radius bayangan. Radius bayangan
    // terikat lebar penumbra supaya bayangan kontak tetap tajam; AO tidak punya
    // penumbra dan justru butuh peredaman guratan arah irisan GTAO.
    constexpr float AO_BLUR_RADIUS_PX = 4.0f;

    // ---- VOLUMETRIC LIGHT SHAFT ------------------------------------------
    // Sinar matahari yang terlihat karena udara menghamburkan cahaya. Item
    // terakhir antrean bayangan, dan satu-satunya yang membayangi RUANG alih-alih
    // permukaan.
    //
    // Dihitung di SETENGAH resolusi: hamburan volume berfrekuensi sangat rendah,
    // jadi tidak ada detail yang bisa hilang, sementara biayanya turun 4x.
    // Pembesarannya gratis — main.frag menyampelnya dengan sampler linear.
    //
    // VOL_DENSITY = 0 mematikan fitur ini sepenuhnya.
    // v70: 0,012 -> 0,0021.
    //
    // Angkanya TURUN drastis, dan itu bukan mengecilkan efek — melainkan
    // konsekuensi dua perubahan di shader. Dengan bentuk berbatas
    // (1 - exp(-d*L)) dan fase g=0,78, kerapatan lama akan memberi puncak
    // berkas 6,2x lantai terang: langsung putih lagi.
    //
    // 0,0021 diturunkan dari target: puncak berkas 0,35 HDR di jarak 60 m,
    // yaitu DUA KALI lantai yang kena matahari (0,17). Cukup untuk terlihat
    // jelas sebagai berkas, jauh dari memutihkan gambar.
    constexpr float VOL_DENSITY   = 0.0021f;  // per meter
    // v70: 16 -> 24, batas atas MAX_STEPS di shader.
    //
    // Berkas cahaya TERLIHAT karena kontras antara ruas ray yang tersinari
    // dan yang terhalang. Kalau langkahnya terlalu jarang, struktur bayangan
    // di sepanjang ray tidak pernah tersampel dan yang tersisa cuma nilai
    // rata-rata — persis selubung merata yang kamu lihat.
    //
    //   60 m / 16 langkah = 3,75 m  -> celah antar gedung 1,5 m TIDAK terlihat
    //   60 m / 24 langkah = 2,50 m  -> celah mulai terselesaikan
    constexpr int   VOL_STEPS     = 24;      // maks 24
    constexpr float VOL_MAX_DIST  = 60.0f;   // meter

    // ---- MODE DEBUG ------------------------------------------------------
    // Ubah ke 1/2/3, rebuild, lihat hasilnya. Semua mode mem-BYPASS temporal dan
    // blur supaya yang terlihat benar-benar keluaran mentah csm_resolve.
    //
    //   0 = mati (normal)
    //   1 = INDEKS CASCADE. Empat tingkat kecerahan: cascade 0 paling terang,
    //       cascade 3 paling gelap. Kalau batas antar tingkat jatuh TEPAT di
    //       lokasi "sobekan", berarti penyebabnya perpindahan cascade.
    //   2 = LEBAR PENUMBRA (dinormalisasi ke MAX_PENUMBRA_WORLD).
    //   3 = BAYANGAN MENTAH, tanpa temporal dan tanpa blur. Kalau sobekan tetap
    //       ada di sini, ia lahir di csm_resolve atau di geometri — bukan di
    //       tahap penyaringan.
    //   4 = AO SAJA — bayangan matahari dipaksa nol sehingga yang tampil murni
    //       ambient occlusion. Cara tercepat menilai dan menyetel AO.
    // ---- PENYARINGAN BAYANGAN: MATI SECARA BAWAAN --------------------------
    //
    // Ini bukan perbaikan. Ini mundur ke keadaan yang TERBUKTI bersih.
    //
    // Kamu sudah memastikan sendiri DEBUG_MODE 3 tidak berartefak. Mode itu
    // melewati shadow_temporal dan kedua blur, dan membiarkan sisanya utuh —
    // PCSS, RPDB, AO, contact shadow, volumetric, TAA, tonemap semuanya jalan.
    // Jadi menyalakan bypass yang sama sebagai bawaan memberi gambar bersih
    // hari ini, bukan janji perbaikan berikutnya.
    //
    // Yang hilang: peredaman noise dither dan pelembutan penumbra dari filter.
    // Itu harga yang jauh lebih murah daripada artefak yang tidak hilang-hilang.
    //
    // Kenapa saya berhenti memperbaikinya di sini: setiap tebakan saya harus
    // kamu uji dengan build lambat di rasterizer perangkat lunak, dan filter
    // spasial-temporal justru kelas teknik yang paling tidak bisa dinilai di
    // 0,5 FPS. Nyalakan lagi (true) setelah ada GPU, di mana satu putaran uji
    // hitungan detik dan hasilnya bisa dipercaya.
    // Dinyalakan kembali: penyaringnya sekarang SVGF a-trous, bukan blur
    // separable yang radiusnya berubah menurut posisi.
    constexpr bool  SHADOW_FILTER_ENABLED = true;

    constexpr int   DEBUG_MODE          = 0;

    // AKTIF. Audit strukturnya: tiap sel dinding menghasilkan tepat satu tile,
    // AABB-nya persis sama dengan posisi vertexnya di ketiga sumbu, dan tile
    // menutupi 100% geometri level secara urut menaik — jadi tidak ada geometri
    // yang bisa hilang. Ujinya juga sengaja konservatif: tidak meng-cull di sisi
    // dekat cahaya, sehingga caster di antara cahaya dan cascade tetap tergambar.
    //
    // Kalau perlu dimatikan, ubah ke false: seluruh level digambar ke tiap
    // cascade, sama persis dengan perilaku sebelum fitur ini ada.
    // v48 [1]: MATI. cascadeIntersectsAABB() membuang caster yang proyeksinya
    // di luar [-1,1], dan tidak punya uji untuk hi.z < 0. Caster yang lebih
    // dekat ke matahari daripada near plane cascade DIPOTONG rasterizer dan
    // tidak menulis apa pun ke shadow map — bangunan yang kena kondisi itu
    // tidak punya bayangan sama sekali. Dengan culling mati, tidak ada satu pun
    // caster yang bisa hilang dari cascade mana pun. Mahal, tapi kebocoran
    // tidak mungkin bertahan karena sebab ini.
    constexpr bool  CASCADE_CULLING     = true;

    // ---- SHADOW MAP CACHING (v40) -----------------------------------------
    //
    // Sampai v39 keenam cascade dirender ULANG setiap frame: 6 x 2048^2 =
    // 25,2 juta texel di-clear lalu diraster, selamanya. Profiler engine ini
    // sendiri sudah menunjukkan biaya pass cascade nyaris seluruhnya CLEAR,
    // bukan rasterisasi — jadi yang dibayar adalah kerja yang hasilnya
    // seringkali identik dengan frame sebelumnya.
    //
    // Aturan cache di sini SENGAJA dibuat sesempit mungkin supaya benar tanpa
    // heuristik: sebuah cascade dipakai ulang HANYA kalau matriksnya keluar
    // bit-identik dengan yang dipakai saat ia terakhir dirender, DAN tidak ada
    // caster bergerak yang menyentuh kotaknya. Kalau dua syarat itu terpenuhi,
    // isi shadow map-nya pasti sama persis — tidak ada yang perlu ditebak.
    //
    // CASCADE_CACHE_SNAP_TEXELS memperluas manfaatnya ke saat pemain BERGERAK.
    // Pusat cascade dikunci ke kisi q texel, bukan 1 texel, sehingga matriksnya
    // baru berubah setelah kamera bergeser sejauh q texel. Radiusnya diberi
    // padding (q+1) texel supaya bola pembatas frustum tetap tercakup meski
    // pusatnya bergeser sampai q texel — itu syarat kebenarannya, dan
    // dibuktikan ulang oleh verify_v40.py.
    //
    // Cascade dekat sengaja q=1 (perilaku lama persis): texelnya 0,2 cm, jadi
    // menaikkan q tidak menghemat apa-apa sementara padding-nya tetap memakan
    // resolusi di tempat yang paling butuh.
    // v48 [2]: MATI sementara. Cache memakai ulang layer selama matriksnya
    // bit-identik — logikanya benar, tapi ia satu-satunya jalur di engine ini
    // yang bisa menampilkan shadow map dari frame LAIN. Selama sumber
    // kebocoran belum dipastikan, jalur itu tidak boleh ikut jadi variabel.
    // Nyalakan lagi begitu kebocorannya beres.
    constexpr bool  CASCADE_CACHE       = true;
    constexpr uint32_t CASCADE_CACHE_SNAP_TEXELS[6] = {1, 1, 2, 4, 8, 16};

    // ---- CULLING CASTER KECIL (v43) ---------------------------------------
    //
    // Caster yang lebarnya kurang dari sekian texel di sebuah cascade tidak
    // digambar ke cascade itu. Ini teknik AAA standar, dan di engine ini ia
    // menyelesaikan dua hal sekaligus.
    //
    // Bola berjari-jari 0,35 m tebalnya, menurut texel dari laporan cascade:
    //   C0 233  C1 140  C2 78  C3 39  C4 17  C5 5,8 texel
    //
    // Di C5 ia 5,8 texel — bayangannya sudah tidak berbentuk apa pun, tapi
    // kehadirannya memaksa cascade 2048^2 itu dirender ulang SETIAP FRAME.
    // Dengan ambang 24, bola berhenti digambar ke C4 dan C5, dan kedua
    // cascade itu langsung memenuhi syarat cache.
    //
    // Ambangnya jangan dinaikkan sembarangan: 100 akan membuang bola dari C2
    // juga, padahal di sana ia masih 78 texel — bayangannya jelas terlihat.
    constexpr float MIN_CASTER_TEXELS   = 24.0f;

    // Depth bias pipeline shadow pass — sengaja kecil; kerja utama anti-acne
    // dilakukan oleh normal-offset bias di csm_resolve.comp.
    // v47: skala offset sisi caster. Dulu terkunci 1,0 texel di dalam shader.
    // Diangkat jadi konstanta supaya bisa disetel bersama DEPTH_BIAS_MAX_WORLD
    // — keduanya menggeser bayangan ke arah yang sama, jadi menyetel satu tanpa
    // yang lain cuma memindahkan masalahnya.
    constexpr float CASTER_OFFSET_SCALE     = 0.35f;

    constexpr float RASTER_DEPTH_BIAS_CONST = 1.0f;
    constexpr float RASTER_DEPTH_BIAS_SLOPE = 1.75f;
}

// ============================================================================
// UBO — layout HARUS identik dengan blok ShadowUBO di semua .comp
// ============================================================================
struct ShadowUBO {
    glm::mat4 cascadeVP[6];
    glm::mat4 invViewProj;
    glm::mat4 prevViewProj;
    glm::mat4 view;
    glm::mat4 viewProj;            // BARU: memproyeksikan titik ray contact shadow
    // Skalar per-cascade. Dengan 6 cascade tidak muat lagi di satu vec4, jadi
    // vec4[2] (8 slot, 2 terbuang). Diindeks [c>>2][c&3] di C++ maupun GLSL.
    glm::vec4 cascadeSplitView[2];
    glm::vec4 cascadeExtentWorld[2];
    glm::vec4 cascadeDepthRange[2];
    glm::vec4 cascadeTexelWorld[2];
    glm::vec4 lightDirWorld;
    glm::vec4 cameraPos;      // xyz = posisi, w = tan(fovY/2)
    glm::vec4 resolution;     // xy = res, zw = 1/res
    glm::vec4 params0;        // x=sunTan y=minPenumbra z=maxPenumbra w=normalOffsetScale
    glm::vec4 params1;        // x=depthBiasScale y=temporalBlend z=disoccRel w=blurMaxRadiusPx
    glm::vec4 params2;        // x=frameIndex y=nearZ z=farZ w=cascadeBlendFrac
    glm::vec4 params3;        // x=shadowRes y=1/shadowRes z=blockerSearchWorld w=tauStatic
    glm::vec4 params4;        // x=deltaTime y=minBlurRadiusPx z=minPcfScreenPx w=pakaiBilinearManual
    glm::vec4 params5;        // x=debugMode y=jarakBayanganEfektif z=fadeFrac
    glm::vec4 params6;        // contact shadow: x=panjang(m) y=jumlahLangkah z=tebal(m) w=jarakFade(m)
    glm::vec4 params7;        // AO: x=radius(m) y=slices z=steps w=intensitas
    glm::vec4 params8;        // x=aoMinVisibility y=aoBlurRadiusPx z=aoMaxRadiusPx
    glm::vec4 params9;        // x=normalOffsetMaxWorld y=depthBiasMaxWorld
};
static_assert(sizeof(ShadowUBO) % 16 == 0, "ShadowUBO harus kelipatan 16 byte (std140)");

// ============================================================================
// GpuProfiler — timestamp query per pass
//
//   Tujuannya satu: berhenti menebak ke mana waktu GPU pergi. Sebelum ada ini,
//   satu-satunya angka yang kita punya adalah FPS hasil menghitung frame unik
//   di rekaman layar (1-17 FPS) — cukup untuk tahu ada masalah, tidak cukup
//   untuk tahu HARUS mengoptimasi apa.
//
//   Cara kerja: satu query pool per slot frame. Tiap pass menulis satu timestamp
//   di ujungnya (eBottomOfPipe = "semua sebelum ini sudah selesai"), lalu durasi
//   tiap pass = selisih antar timestamp berurutan.
//
//   Hasilnya dibaca dari slot yang fence-nya SUDAH ditunggu di drawFrame, jadi
//   pembacaannya tidak pernah menghentikan GPU.
// ============================================================================
class GpuProfiler {
public:
    enum Marker {
        FrameStart = 0, Prepass, Cascades, Resolve, Temporal,
        BlurH, BlurV, MainPass, UiPass, MARKER_COUNT
    };

    static const char* markerName(int i) {
        switch (i) {
            case Prepass:   return "prepass";
            case Cascades:  return "cascade x4";
            case Resolve:   return "pcss resolve";
            case Temporal:  return "temporal";
            case BlurH:     return "blur H";
            case BlurV:     return "blur V";
            case MainPass:  return "main pass";
            case UiPass:    return "ui";
            default:        return "?";
        }
    }

    void init(vk::PhysicalDevice pd, vk::Device dev, uint32_t frames) {
        m_device = dev;
        m_frames = std::max(1u, frames);

        auto props = pd.getProperties();
        m_nsPerTick = props.limits.timestampPeriod;

        // timestampComputeAndGraphics = semua queue graphics DAN compute
        // mendukung timestamp. Kalau false, dukungannya per queue family dan
        // belum tentu ada; lebih aman menonaktifkan profiler daripada membaca
        // angka sampah.
        m_enabled = (props.limits.timestampComputeAndGraphics == VK_TRUE) && (m_nsPerTick > 0.0f);
        if (!m_enabled) return;

        m_pools.resize(m_frames);
        m_written.assign(m_frames, false);
        for (uint32_t i = 0; i < m_frames; ++i) {
            vk::QueryPoolCreateInfo qi{};
            qi.queryType  = vk::QueryType::eTimestamp;
            qi.queryCount = MARKER_COUNT;
            m_pools[i] = m_device.createQueryPool(qi);
        }
        m_accum.assign(MARKER_COUNT, 0.0);
        m_lastReport = std::chrono::steady_clock::now();
    }

    void destroy() {
        if (!m_device) return;
        for (auto p : m_pools) if (p) m_device.destroyQueryPool(p);
        m_pools.clear();
        m_device = nullptr;
    }

    bool enabled() const { return m_enabled; }

    // Dipanggil sekali di awal command buffer, DI LUAR render pass.
    void beginFrame(vk::CommandBuffer cmd, uint32_t slot) {
        if (!m_enabled) return;
        cmd.resetQueryPool(m_pools[slot], 0, MARKER_COUNT);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, m_pools[slot], FrameStart);
        m_written[slot] = true;
    }

    void stamp(vk::CommandBuffer cmd, uint32_t slot, Marker m) {
        if (!m_enabled) return;
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pools[slot], m);
    }

    // Dipanggil SETELAH fence slot ini ditunggu di drawFrame — hasilnya dijamin
    // sudah siap, jadi tidak ada stall.
    void collect(uint32_t slot) {
        if (!m_enabled || !m_written[slot]) return;

        uint64_t ts[MARKER_COUNT] = {};
        vk::Result r = m_device.getQueryPoolResults(
            m_pools[slot], 0, MARKER_COUNT, sizeof(ts), ts, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64);
        if (r != vk::Result::eSuccess) return;

        for (int i = 1; i < MARKER_COUNT; ++i) {
            // Timestamp bisa mundur kalau query tidak terisi; buang sampahnya.
            if (ts[i] < ts[i - 1]) return;
            m_accum[i] += double(ts[i] - ts[i - 1]) * double(m_nsPerTick) * 1e-6; // ms
        }
        ++m_samples;
    }

    bool hasReport() const { return m_enabled && m_samples >= SAMPLES_PER_REPORT; }

    std::string takeReport() {
        auto now  = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double>(now - m_lastReport).count();
        m_lastReport = now;

        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        double total = 0.0;
        for (int i = 1; i < MARKER_COUNT; ++i) {
            double ms = m_accum[i] / double(m_samples);
            total += ms;
            os << markerName(i) << " " << ms << " | ";
        }
        os << "GPU total " << total << " ms";
        if (wall > 0.0) {
            os << "  ||  FPS nyata " << (double(m_samples) / wall)
               << "  (frame " << (wall * 1000.0 / double(m_samples)) << " ms)";
        }

        std::fill(m_accum.begin(), m_accum.end(), 0.0);
        m_samples = 0;
        return os.str();
    }

private:
    static constexpr int SAMPLES_PER_REPORT = 15;

    vk::Device                   m_device;
    std::vector<vk::QueryPool>   m_pools;
    std::vector<bool>            m_written;
    std::vector<double>          m_accum;
    uint32_t                     m_frames    = 1;
    int                          m_samples   = 0;
    float                        m_nsPerTick = 0.0f;
    bool                         m_enabled   = false;
    std::chrono::steady_clock::time_point m_lastReport;
};

// ============================================================================
// INPUT PER FRAME
// ============================================================================
struct FrameInput {
    glm::vec3 cameraPos     {0.0f};
    glm::vec3 cameraForward {0.0f, 0.0f, -1.0f};
    glm::vec3 cameraRight   {1.0f, 0.0f, 0.0f};
    glm::vec3 cameraUp      {0.0f, 1.0f, 0.0f};
    float     fovYRadians   = glm::radians(45.0f);
    float     aspect        = 16.0f / 9.0f;
    float     nearZ         = 0.1f;
    float     farZ          = 100.0f;

    // Jarak yang harus dicakup cascade. TERPISAH dari farZ dengan sengaja:
    // farZ juga mengisi params2.z, yang dipakai linearizeDepth() di shader —
    // mengecilkannya akan merusak linearisasi kedalaman di seluruh pipeline.
    // Nilai 0 berarti "pakai SHADOW_MAX_DISTANCE seperti dulu".
    float     shadowFitDist = 0.0f;

    // v40: kotak pembatas SEMUA caster yang bisa bergerak, digabung jadi satu.
    // Cascade yang berpotongan dengan kotak ini tidak boleh di-cache — isinya
    // berubah walau matriksnya tidak. Kotaknya boleh (dan sebaiknya) konservatif:
    // untuk benda yang bergerak berulang, pakai selubung SELURUH lintasannya
    // sekali saja, bukan posisi sesaat, supaya tidak perlu diperbarui tiap frame.
    // hasDynamic = false berarti seluruh scene statis.
    bool      hasDynamic    = false;
    glm::vec3 dynamicMin    {0.0f};
    glm::vec3 dynamicMax    {0.0f};
    // v43: jari-jari caster dinamis terbesar. Dipakai untuk culling caster
    // kecil — caster yang lebih tipis dari MIN_CASTER_TEXELS di sebuah cascade
    // tidak digambar ke sana, jadi ia juga tidak boleh membatalkan cache-nya.
    float     dynamicRadius = 0.0f;

    glm::vec3 lightDir      {0.0f, 1.0f, 0.0f}; // arah MENUJU cahaya
    glm::mat4 view          {1.0f};
    glm::mat4 proj          {1.0f};             // sudah dengan proj[1][1] *= -1
    float     deltaTime     = 1.0f / 60.0f;     // detik; dipakai blending temporal
    // Diisi aplikasi sejak TAA aktif: viewProj frame lalu yang sudah dijitter
    // dengan jitter frame INI, supaya velocity keluar bebas jitter.
    glm::mat4 prevViewProj  {1.0f};
    bool      hasPrevVP     = true;
};

struct InitInfo {
    vk::PhysicalDevice physicalDevice;
    vk::Device         device;
    vk::Queue          graphicsQueue;
    vk::CommandPool    commandPool;      // untuk one-time layout transition
    vk::Extent2D       screenExtent;
    uint32_t           framesInFlight = 2;
    std::string        shaderDir;        // prefix path, mis. "" atau "shaders/"

    // Deskripsi vertex input milik aplikasi, supaya modul ini tidak perlu
    // tahu tentang struct Vertex di main2.cpp.
    vk::VertexInputBindingDescription                  vertexBinding;
    std::vector<vk::VertexInputAttributeDescription>   vertexAttributes;
};

// cascadeIndex: -1 = depth pre-pass (gambar semua yang terlihat kamera),
//               0..NUM_CASCADES-1 = pass cascade (boleh di-cull per cascade).
using DrawSceneFn = std::function<void(vk::CommandBuffer, int cascadeIndex)>;

// ============================================================================
class Pipeline {
public:
    void init(const InitInfo& info);
    void destroy();

    // Dipanggil dari worker thread (updateGameState). Menghitung cascade,
    // lalu menulis UBO langsung ke slot yang bersangkutan. Aman dipanggil
    // paralel dengan record() slot LAIN.
    void updateCascades(const FrameInput& in, uint32_t frameSlot);

    // Dipanggil dari thread perekam command buffer, SEBELUM main render pass.
    // prof boleh nullptr; kalau diisi, tiap pass menulis timestamp di ujungnya.
    // v43: apakah caster berjari-jari r layak digambar ke cascade c?
    // SATU-SATUNYA sumber kebenaran untuk aturan ini — dipakai baik oleh
    // drawScene (melewatkan draw) maupun oleh keputusan cache.
    bool casterVisibleInCascade(int c, float radiusWorld) const {
        if (c < 0 || c >= static_cast<int>(Cfg::NUM_CASCADES)) return true;
        const float t = m_lastTexelWorld[static_cast<size_t>(c)];
        if (t <= 0.0f) return true;                       // belum terisi
        return (2.0f * radiusWorld / t) >= Cfg::MIN_CASTER_TEXELS;
    }

    void record(vk::CommandBuffer cmd, uint32_t frameSlot, const DrawSceneFn& drawScene,
                GpuProfiler* prof = nullptr);

    // ---- Uji culling per-cascade -------------------------------------------
    // Mengembalikan false kalau AABB dunia ini pasti tidak menyumbang bayangan
    // apa pun ke cascade tersebut. Aman dipanggil dari lambda drawScene.
    //
    // Sengaja TIDAK meng-cull di sisi dekat cahaya (lo.z < 0): objek yang berada
    // antara cahaya dan cascade tetap melemparkan bayangan ke dalamnya. Yang
    // dibuang hanya yang di luar kotak XY, atau yang seluruhnya di belakang far
    // plane cascade.
    bool cascadeIntersectsAABB(uint32_t frameSlot, uint32_t cascadeIndex,
                               const glm::vec3& mn, const glm::vec3& mx) const {
        if (frameSlot >= m_slots.size() || cascadeIndex >= Cfg::NUM_CASCADES) return true;
        const glm::mat4& vp = m_slots[frameSlot].cascadeVP[cascadeIndex];

        glm::vec3 lo( 1e30f), hi(-1e30f);
        for (int i = 0; i < 8; ++i) {
            glm::vec4 p = vp * glm::vec4((i & 1) ? mx.x : mn.x,
                                         (i & 2) ? mx.y : mn.y,
                                         (i & 4) ? mx.z : mn.z, 1.0f);
            lo = glm::min(lo, glm::vec3(p));
            hi = glm::max(hi, glm::vec3(p));
        }
        if (hi.x < -1.0f || lo.x > 1.0f) return false;
        if (hi.y < -1.0f || lo.y > 1.0f) return false;
        if (lo.z >  1.0f)                return false;
        return true;
    }

    // Yang di-bind main pass di binding = 1 (menggantikan shadow map lama)
    vk::ImageView   finalShadowView()    const { return m_shadowFinal.view; }
    vk::Sampler     finalShadowSampler() const { return m_samplerScreen; }
    vk::ImageLayout finalShadowLayout()  const { return vk::ImageLayout::eGeneral; }

    // true  = 2x2 hardware PCF tersedia (jalur murah)
    // false = shader memakai bilinear manual, 4x fetch per tap
    bool hardwarePcfAvailable() const { return m_cmpLinearOK; }

    // Light shaft, setengah resolusi. Dibaca main pass di binding 2 dengan
    // sampler linear, jadi pembesarannya ditangani kartu grafis.
    vk::ImageView   volumetricView()    const { return m_volumetric.view; }
    vk::Sampler     volumetricSampler() const { return m_samplerScreen; }
    vk::ImageLayout volumetricLayout()  const { return vk::ImageLayout::eGeneral; }

    // Dipakai TAA: motion vector dan depth dari pre-pass. Keduanya sudah ada
    // sejak v1 — TAA tinggal memakainya, tidak perlu pass baru.
    vk::ImageView velocityView()     const { return m_velocity.view; }
    vk::ImageView prepassDepthView() const { return m_prepassDepth.view; }

    // Laporan sekali jalan: jangkauan tiap cascade, ukuran texel-nya, dan
    // rasio texel-per-piksel. Rasio itu ukuran mutu yang sesungguhnya — selama
    // ia sekitar 1, ketajaman yang TERLIHAT sama di semua cascade, seberapa pun
    // jauh jangkauannya. Rasio jauh di atas 1 berarti cascade itu kurang
    // resolusi; jauh di bawah 1 berarti resolusinya terbuang.
    std::string cascadeReport() const {
        if (!m_hasStats) return "cascade belum dihitung";
        std::string r = "jangkauan efektif " + fmt1(m_statSplit[Cfg::NUM_CASCADES - 1]) + " m";
        float prev = 0.0f;
        for (uint32_t c = 0; c < Cfg::NUM_CASCADES; ++c) {
            // Ukuran dunia satu piksel layar di tengah rentang cascade ini
            float mid  = 0.5f * (prev + m_statSplit[c]);
            float wpp  = 2.0f * mid * m_statTanHalfFovY / m_statResY;
            r += " | C" + std::to_string(c) + " " + fmt1(prev) + "-" + fmt1(m_statSplit[c])
               + "m texel " + fmt1(m_statTexel[c] * 100.0f) + "cm rasio "
               + fmt1(m_statTexel[c] / std::max(wpp, 1e-6f));
            prev = m_statSplit[c];
        }
        return r;
    }

private:
    // ---------------------------------------------------------------- helpers
    struct Img {
        vk::Image        image;
        vk::DeviceMemory memory;
        vk::ImageView    view;
    };

    struct SlotData {
        glm::mat4 cascadeVP[Cfg::NUM_CASCADES];
        glm::mat4 viewProj;
        glm::mat4 prevViewProj;
        // Dibutuhkan pass cascade: normal offset kini dikerjakan di sisi CASTER,
        // jadi shader-nya perlu tahu arah cahaya dan ukuran texel cascade ini.
        float     texelWorld[Cfg::NUM_CASCADES]{};
        glm::vec3 lightDir{0.0f, 1.0f, 0.0f};
        uint32_t  parity = 0;
        // v40 shadow map caching: false = layer ini TIDAK dirender ulang frame
        // ini, isinya dipakai apa adanya dari frame sebelumnya.
        bool      cascadeNeedsRender[Cfg::NUM_CASCADES]{};
    };

    static std::string fmt1(float v) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.1f", static_cast<double>(v));
        return std::string(b);
    }

    static std::vector<char> readFile(const std::string& path) {
        bool ok = false;
        auto buf = platformLoadFile(path, &ok);          // v73: aset APK juga
        if (!ok) throw std::runtime_error("ShadowAAA: gagal buka " + path);
        return buf;
    }

    vk::ShaderModule loadShader(const std::string& name) {
        auto code = readFile(m_shaderDir + name);
        vk::ShaderModuleCreateInfo ci{};
        ci.codeSize = code.size();
        ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        return m_device.createShaderModule(ci);
    }

    uint32_t findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) {
        auto mp = m_physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("ShadowAAA: tidak ada memory type yang cocok");
    }

    Img createImage(vk::Format fmt, vk::Extent2D ext, uint32_t layers,
                    vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
                    vk::ImageViewType viewType) {
        Img out{};
        vk::ImageCreateInfo ii{};
        ii.imageType     = vk::ImageType::e2D;
        ii.extent        = vk::Extent3D{ext.width, ext.height, 1};
        ii.mipLevels     = 1;
        ii.arrayLayers   = layers;
        ii.format        = fmt;
        ii.tiling        = vk::ImageTiling::eOptimal;
        ii.initialLayout = vk::ImageLayout::eUndefined;
        ii.usage         = usage;
        ii.samples       = vk::SampleCountFlagBits::e1;
        ii.sharingMode   = vk::SharingMode::eExclusive;
        out.image = m_device.createImage(ii);

        auto mr = m_device.getImageMemoryRequirements(out.image);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                            vk::MemoryPropertyFlagBits::eDeviceLocal);
        out.memory = m_device.allocateMemory(ai);
        m_device.bindImageMemory(out.image, out.memory, 0);

        vk::ImageViewCreateInfo vi{};
        vi.image            = out.image;
        vi.viewType         = viewType;
        vi.format           = fmt;
        vi.subresourceRange = vk::ImageSubresourceRange{aspect, 0, 1, 0, layers};
        out.view = m_device.createImageView(vi);
        return out;
    }

    void destroyImage(Img& i) {
        if (i.view)   m_device.destroyImageView(i.view);
        if (i.image)  m_device.destroyImage(i.image);
        if (i.memory) m_device.freeMemory(i.memory);
        i = Img{};
    }

    static void computeBarrier(vk::CommandBuffer cmd,
                               vk::PipelineStageFlags dstStage,
                               vk::AccessFlags dstAccess) {
        vk::MemoryBarrier mb{};
        mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        mb.dstAccessMask = dstAccess;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            dstStage, {}, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // ---------------------------------------------------------------- create
    void createResources(const InitInfo& info);
    void createBlueNoiseLUT(const InitInfo& info);
    void createSamplers();
    void createRenderPasses();
    void createFramebuffers();
    void createGraphicsPipelines(const InitInfo& info);
    void createComputePipelines();
    void createDescriptors();
    void transitionStorageImages(const InitInfo& info);

    // ---------------------------------------------------------------- state
    vk::PhysicalDevice m_physicalDevice;
    vk::Device         m_device;
    vk::Extent2D       m_screenExtent{};
    vk::Extent2D       m_cascadeExtent{Cfg::CASCADE_RES, Cfg::CASCADE_RES};
    uint32_t           m_framesInFlight = 2;
    std::string        m_shaderDir;
    vk::Format         m_depthFormat = vk::Format::eD32Sfloat;
    // Apakah driver mendukung filter linear (2x2 hardware PCF) pada format depth
    // cascade. Kalau tidak, shader memakai bilinear manual.
    bool               m_cmpLinearOK = false;

    // Images
    Img m_prepassDepth{}, m_gNormal{}, m_velocity{};
    Img m_cascades{};                                  // texture array 4 layer
    std::array<vk::ImageView, Cfg::NUM_CASCADES> m_cascadeLayerViews{};
    // m_svgfA / m_svgfB: ping-pong antar iterasi a-trous (dulu m_blurTmp).
       // m_moments[2]: momen ke-1 dan ke-2 + panjang riwayat untuk variansi SVGF.
    Img m_shadowResolve{}, m_svgfA{}, m_shadowFinal{}, m_aoRaw{}, m_volumetric{}, m_svgfB{};
    // LUT blue noise 128x128 R8_UNORM, sampler REPEAT (tile-nya toroidal).
    Img          m_blueNoise{};
    vk::Sampler  m_samplerBlueNoise{};
    std::array<Img, 2> m_moments{};
    vk::Extent2D m_volExtent{};
    std::array<Img, 2> m_history{};

    // Samplers
    vk::Sampler m_samplerScreen{};        // linear clamp untuk texture layar
    vk::Sampler m_samplerCascade{};       // nearest clamp-to-border (blocker search)
    vk::Sampler m_samplerCascadeCmp{};    // compare LEQUAL (hardware PCF)

    // Render passes & framebuffers
    // v40: matriks yang BENAR-BENAR dipakai saat tiap layer terakhir dirender.
    // Bukan per frame-slot: shadow map array-nya sendiri dipakai bersama semua
    // frame in flight, jadi cache-nya juga harus tunggal.
    // v43: texelWorld terakhir per cascade, dipakai accessor culling caster.
    std::array<float, Cfg::NUM_CASCADES> m_lastTexelWorld{};

    std::array<glm::mat4, Cfg::NUM_CASCADES> m_cachedCascadeVP{};
    std::array<bool,      Cfg::NUM_CASCADES> m_cascadeEverRendered{};
    uint64_t m_cascadeRenderCount = 0, m_cascadeSkipCount = 0;
    std::array<uint64_t, Cfg::NUM_CASCADES> m_cascadeSkipPer{};

    vk::RenderPass  m_prepassRP{}, m_cascadeRP{};
    vk::Framebuffer m_prepassFB{};
    std::array<vk::Framebuffer, Cfg::NUM_CASCADES> m_cascadeFB{};

    // Graphics pipelines
    vk::PipelineLayout m_prepassLayout{}, m_cascadeLayout{};
    vk::Pipeline       m_prepassPipeline{}, m_cascadePipeline{};

    // Compute pipelines
    vk::DescriptorSetLayout m_resolveDSL{}, m_temporalDSL{}, m_atrousDSL{}, m_aoDSL{}, m_volDSL{};
    vk::PipelineLayout      m_resolvePL{},  m_temporalPL{},  m_atrousPL{},  m_aoPL{},  m_volPL{};
    vk::Pipeline            m_resolvePipe{}, m_temporalPipe{},
                            m_atrousPipe{}, m_aoPipe{}, m_volPipe{};

    // Descriptors
    vk::DescriptorPool m_descPool{};
    std::vector<vk::DescriptorSet>                m_resolveSets;   // [slot]
    std::vector<std::array<vk::DescriptorSet, 2>> m_temporalSets;  // [slot][parity]
    // [slot][parity] untuk iterasi 0 (masukannya history yang ping-pong),
    // lalu tiga iterasi berikutnya yang rantainya tetap.
    std::vector<std::array<vk::DescriptorSet, 2>> m_atrous0Sets;
    std::vector<std::array<vk::DescriptorSet, 3>> m_atrousNSets;
    std::vector<vk::DescriptorSet>                m_aoSets;        // [slot]
    std::vector<vk::DescriptorSet>                m_volSets;       // [slot]

    // UBO
    std::vector<vk::Buffer>       m_uboBuffers;
    std::vector<vk::DeviceMemory> m_uboMemory;
    std::vector<void*>            m_uboMapped;

    // Per-slot CPU state
    std::vector<SlotData> m_slots;

    // Statistik cascade untuk laporan diagnostik
    std::array<float, Cfg::NUM_CASCADES> m_statSplit{}, m_statExtent{}, m_statTexel{};
    float m_statTanHalfFovY = 0.0f, m_statResY = 1.0f;
    bool  m_hasStats = false;

    // Sequential state
    glm::mat4 m_lastViewProj{1.0f};
    bool      m_hasHistory   = false;
    uint32_t  m_frameCounter = 0;
};

// ============================================================================
// IMPLEMENTASI
// ============================================================================

inline void Pipeline::init(const InitInfo& info) {
    m_physicalDevice = info.physicalDevice;
    m_device         = info.device;
    m_screenExtent   = info.screenExtent;
    m_framesInFlight = std::max(1u, info.framesInFlight);
    m_shaderDir      = info.shaderDir;

    // Pilih format depth yang didukung.
    // Sengaja HANYA format depth-only: format ber-stencil mengharuskan image view
    // attachment memuat aspect stencil juga, dan itu bentrok dengan view
    // depth-only yang kita pakai untuk sampling.
    for (vk::Format f : {vk::Format::eD32Sfloat, vk::Format::eX8D24UnormPack32,
                         vk::Format::eD16Unorm}) {
        auto p = m_physicalDevice.getFormatProperties(f);
        if (p.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            m_depthFormat = f;
            break;
        }
    }

    m_slots.resize(m_framesInFlight);

    createResources(info);
    createSamplers();
    createRenderPasses();
    createFramebuffers();
    createGraphicsPipelines(info);
    createComputePipelines();
    createDescriptors();
    transitionStorageImages(info);
}

// ---------------------------------------------------------------------------
inline void Pipeline::createBlueNoiseLUT(const InitInfo& info) {
    const vk::DeviceSize sz = 128 * 128;
    m_blueNoise = createImage(vk::Format::eR8Unorm, vk::Extent2D{128, 128}, 1,
                              vk::ImageUsageFlagBits::eSampled |
                              vk::ImageUsageFlagBits::eTransferDst,
                              vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);

    vk::BufferCreateInfo bi{};
    bi.size = sz; bi.usage = vk::BufferUsageFlagBits::eTransferSrc;
    vk::Buffer stg = m_device.createBuffer(bi);
    auto mr = m_device.getBufferMemoryRequirements(stg);
    vk::MemoryAllocateInfo mai{};
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stgMem = m_device.allocateMemory(mai);
    m_device.bindBufferMemory(stg, stgMem, 0);
    void* dst = m_device.mapMemory(stgMem, 0, sz);
    std::memcpy(dst, BLUE_NOISE_128, static_cast<size_t>(sz));
    m_device.unmapMemory(stgMem);

    vk::CommandBufferAllocateInfo cai{};
    cai.level = vk::CommandBufferLevel::ePrimary;
    cai.commandPool = info.commandPool; cai.commandBufferCount = 1;
    auto cb = m_device.allocateCommandBuffers(cai)[0];
    vk::CommandBufferBeginInfo cbi{};
    cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cb.begin(cbi);

    vk::ImageMemoryBarrier b{};
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_blueNoise.image;
    b.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    b.oldLayout = vk::ImageLayout::eUndefined;
    b.newLayout = vk::ImageLayout::eTransferDstOptimal;
    b.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1, &b);

    vk::BufferImageCopy rg{};
    rg.imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    rg.imageExtent = vk::Extent3D{128, 128, 1};
    cb.copyBufferToImage(stg, m_blueNoise.image,
                         vk::ImageLayout::eTransferDstOptimal, 1, &rg);

    b.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    b.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    b.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    b.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                       vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 0, nullptr, 1, &b);

    cb.end();
    vk::SubmitInfo si{}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    info.graphicsQueue.submit(si, VK_NULL_HANDLE);
    info.graphicsQueue.waitIdle();
    m_device.freeCommandBuffers(info.commandPool, cb);
    m_device.destroyBuffer(stg);
    m_device.freeMemory(stgMem);

    // REPEAT: tile blue noise bersifat toroidal, jadi tidak ada seam.
    vk::SamplerCreateInfo sci{};
    sci.magFilter = vk::Filter::eNearest;
    sci.minFilter = vk::Filter::eNearest;
    sci.addressModeU = vk::SamplerAddressMode::eRepeat;
    sci.addressModeV = vk::SamplerAddressMode::eRepeat;
    sci.addressModeW = vk::SamplerAddressMode::eRepeat;
    sci.mipmapMode   = vk::SamplerMipmapMode::eNearest;
    m_samplerBlueNoise = m_device.createSampler(sci);
}

inline void Pipeline::createResources(const InitInfo& info) {
    using U = vk::ImageUsageFlagBits;
    const vk::Extent2D S = m_screenExtent;

    m_prepassDepth = createImage(m_depthFormat, S, 1,
                                 U::eDepthStencilAttachment | U::eSampled,
                                 vk::ImageAspectFlagBits::eDepth,
                                 vk::ImageViewType::e2D);

    m_gNormal  = createImage(vk::Format::eR16G16Sfloat, S, 1,
                             U::eColorAttachment | U::eSampled,
                             vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);

    m_velocity = createImage(vk::Format::eR16G16Sfloat, S, 1,
                             U::eColorAttachment | U::eSampled,
                             vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);

    // Cascade array + view per layer untuk framebuffer
    m_cascades = createImage(m_depthFormat, m_cascadeExtent, Cfg::NUM_CASCADES,
                             U::eDepthStencilAttachment | U::eSampled,
                             vk::ImageAspectFlagBits::eDepth,
                             vk::ImageViewType::e2DArray);

    for (uint32_t i = 0; i < Cfg::NUM_CASCADES; ++i) {
        vk::ImageViewCreateInfo vi{};
        vi.image            = m_cascades.image;
        vi.viewType         = vk::ImageViewType::e2D;
        vi.format           = m_depthFormat;
        vi.subresourceRange = vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eDepth, 0, 1, i, 1};
        m_cascadeLayerViews[i] = m_device.createImageView(vi);
    }

    // eTransferDst wajib ada: transitionStorageImages() menolkan isi image ini
    // lewat vkCmdClearColorImage, dan clear termasuk operasi transfer.
    const auto SW = U::eStorage | U::eSampled | U::eTransferDst;
    m_shadowResolve = createImage(vk::Format::eR16G16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    m_history[0]    = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    m_history[1]    = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    m_svgfA         = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    m_svgfB         = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    // Momen SVGF: .r m2_shadow .g m2_ao .b panjang riwayat .a variansi AO
    m_moments[0]    = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    m_moments[1]    = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    // rgba16f: .r bayangan, .g AO, .ba bent normal (oct). Dua kanal terakhir
    // dulu kosong, jadi bent normal masuk tanpa image maupun binding baru di
    // render pass utama.
    m_shadowFinal   = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);
    // AO mentah + bent normal, sebelum penyaringan.
    // .r AO, .gb arah bent normal (oct), .a cadangan.
    m_aoRaw         = createImage(vk::Format::eR16G16B16A16Sfloat, S, 1, SW,
                                  vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);

    // ---- LUT blue noise -----------------------------------------------------
    // Menggantikan interleaved gradient noise. IGN itu murah dan bagus secara
    // temporal, tapi sebaran spasialnya mengandung energi frekuensi rendah —
    // yang tampak sebagai bercak/gumpalan saat sampelnya sedikit. Blue noise
    // memindahkan energinya ke frekuensi tinggi, sehingga sisa noise-nya jauh
    // lebih mudah dihapus filter dan jauh kurang terlihat mata kalau tersisa.
    createBlueNoiseLUT(info);

    // Light shaft di SETENGAH resolusi.
    m_volExtent = vk::Extent2D{(S.width + 1) / 2, (S.height + 1) / 2};
    // rg16f: .r hamburan, .g panjang ray. Kanal kedua dipakai main.frag untuk
    // membedakan texel milik langit dari texel milik permukaan saat membesarkan
    // kembali — tanpa itu, tepi siluet bocor selebar dua piksel.
    m_volumetric = createImage(vk::Format::eR16G16Sfloat, m_volExtent, 1, SW,
                               vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D);

    // UBO per slot
    m_uboBuffers.resize(m_framesInFlight);
    m_uboMemory.resize(m_framesInFlight);
    m_uboMapped.resize(m_framesInFlight);

    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        vk::BufferCreateInfo bi{};
        bi.size        = sizeof(ShadowUBO);
        bi.usage       = vk::BufferUsageFlagBits::eUniformBuffer;
        bi.sharingMode = vk::SharingMode::eExclusive;
        m_uboBuffers[i] = m_device.createBuffer(bi);

        auto mr = m_device.getBufferMemoryRequirements(m_uboBuffers[i]);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_uboMemory[i] = m_device.allocateMemory(ai);
        m_device.bindBufferMemory(m_uboBuffers[i], m_uboMemory[i], 0);
        m_uboMapped[i] = m_device.mapMemory(m_uboMemory[i], 0, sizeof(ShadowUBO));
    }
    (void)info;
}

// ---------------------------------------------------------------------------
inline void Pipeline::createSamplers() {
    // Sampler untuk texture screen-space (history, velocity, resolve, depth)
    {
        vk::SamplerCreateInfo si{};
        si.magFilter = si.minFilter = vk::Filter::eLinear;
        si.addressModeU = si.addressModeV = si.addressModeW =
            vk::SamplerAddressMode::eClampToEdge;
        si.mipmapMode = vk::SamplerMipmapMode::eNearest;
        si.borderColor = vk::BorderColor::eFloatOpaqueBlack;
        m_samplerScreen = m_device.createSampler(si);
    }

    // Blocker search: NEAREST. Filter linear pada format depth tidak dijamin
    // didukung semua GPU (SAMPLED_IMAGE_FILTER_LINEAR opsional untuk D32_SFLOAT),
    // dan blocker search memang tidak diuntungkan oleh interpolasi.
    {
        vk::SamplerCreateInfo si{};
        si.magFilter = si.minFilter = vk::Filter::eNearest;
        si.addressModeU = si.addressModeV = si.addressModeW =
            vk::SamplerAddressMode::eClampToBorder;
        si.borderColor = vk::BorderColor::eFloatOpaqueWhite; // di luar cascade = terang
        si.mipmapMode  = vk::SamplerMipmapMode::eNearest;
        m_samplerCascade = m_device.createSampler(si);
    }

    // PCF: sampler compare. Linear di sini memberi 2x2 hardware PCF gratis.
    //
    // PENTING: kalau driver TIDAK mendukungnya, setiap tap compare jadi biner
    // terhadap satu texel, sehingga posisi tepi bayangan terkunci ke kelipatan
    // texel. Itu muncul sebagai "sobekan": tepi datar sempurna lalu meloncat
    // satu texel sekaligus. Filter selebar apa pun tidak bisa memperbaikinya,
    // karena yang salah posisinya, bukan kelembutannya.
    // Kalau tidak didukung, csm_resolve.comp beralih ke bilinear manual —
    // lihat cmpTap() di sana.
    {
        auto fp = m_physicalDevice.getFormatProperties(m_depthFormat);
        m_cmpLinearOK = static_cast<bool>(
            fp.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);

        vk::SamplerCreateInfo si{};
        si.magFilter = si.minFilter = m_cmpLinearOK ? vk::Filter::eLinear : vk::Filter::eNearest;
        si.addressModeU = si.addressModeV = si.addressModeW =
            vk::SamplerAddressMode::eClampToBorder;
        si.borderColor   = vk::BorderColor::eFloatOpaqueWhite;
        si.compareEnable = VK_TRUE;
        si.compareOp     = vk::CompareOp::eLessOrEqual; // hasil 1.0 = terang
        si.mipmapMode    = vk::SamplerMipmapMode::eNearest;
        m_samplerCascadeCmp = m_device.createSampler(si);
    }
}

// ---------------------------------------------------------------------------
inline void Pipeline::createRenderPasses() {
    // ---- Pre-pass: 2 color (normal, velocity) + depth ----------------------
    {
        std::array<vk::AttachmentDescription, 3> att{};
        for (int i = 0; i < 2; ++i) {
            att[i].format         = vk::Format::eR16G16Sfloat;
            att[i].samples        = vk::SampleCountFlagBits::e1;
            att[i].loadOp         = vk::AttachmentLoadOp::eClear;
            att[i].storeOp        = vk::AttachmentStoreOp::eStore;
            att[i].stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
            att[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            att[i].initialLayout  = vk::ImageLayout::eUndefined;
            att[i].finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
        att[2].format         = m_depthFormat;
        att[2].samples        = vk::SampleCountFlagBits::e1;
        att[2].loadOp         = vk::AttachmentLoadOp::eClear;
        att[2].storeOp        = vk::AttachmentStoreOp::eStore;
        att[2].stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
        att[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        att[2].initialLayout  = vk::ImageLayout::eUndefined;
        att[2].finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal;

        std::array<vk::AttachmentReference, 2> colorRefs{
            vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal},
            vk::AttachmentReference{1, vk::ImageLayout::eColorAttachmentOptimal}};
        vk::AttachmentReference depthRef{2, vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::SubpassDescription sp{};
        sp.pipelineBindPoint       = vk::PipelineBindPoint::eGraphics;
        sp.colorAttachmentCount    = 2;
        sp.pColorAttachments       = colorRefs.data();
        sp.pDepthStencilAttachment = &depthRef;

        std::array<vk::SubpassDependency, 2> deps{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = vk::PipelineStageFlagBits::eComputeShader
                              | vk::PipelineStageFlagBits::eFragmentShader;
        deps[0].srcAccessMask = vk::AccessFlagBits::eShaderRead;
        deps[0].dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput
                              | vk::PipelineStageFlagBits::eEarlyFragmentTests;
        deps[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
                              | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput
                              | vk::PipelineStageFlagBits::eLateFragmentTests;
        deps[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
                              | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        deps[1].dstStageMask  = vk::PipelineStageFlagBits::eComputeShader;
        deps[1].dstAccessMask = vk::AccessFlagBits::eShaderRead;

        vk::RenderPassCreateInfo ci{};
        ci.attachmentCount = 3; ci.pAttachments  = att.data();
        ci.subpassCount    = 1; ci.pSubpasses    = &sp;
        ci.dependencyCount = 2; ci.pDependencies = deps.data();
        m_prepassRP = m_device.createRenderPass(ci);
    }

    // ---- Cascade pass: depth-only -----------------------------------------
    {
        vk::AttachmentDescription da{};
        da.format         = m_depthFormat;
        da.samples        = vk::SampleCountFlagBits::e1;
        da.loadOp         = vk::AttachmentLoadOp::eClear;
        da.storeOp        = vk::AttachmentStoreOp::eStore;
        da.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
        da.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        da.initialLayout  = vk::ImageLayout::eUndefined;
        da.finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal;

        vk::AttachmentReference dr{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::SubpassDescription sp{};
        sp.pipelineBindPoint       = vk::PipelineBindPoint::eGraphics;
        sp.pDepthStencilAttachment = &dr;

        std::array<vk::SubpassDependency, 2> deps{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = vk::PipelineStageFlagBits::eComputeShader
                              | vk::PipelineStageFlagBits::eFragmentShader;
        deps[0].srcAccessMask = vk::AccessFlagBits::eShaderRead;
        deps[0].dstStageMask  = vk::PipelineStageFlagBits::eEarlyFragmentTests
                              | vk::PipelineStageFlagBits::eLateFragmentTests;
        deps[0].dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = vk::PipelineStageFlagBits::eLateFragmentTests;
        deps[1].srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        deps[1].dstStageMask  = vk::PipelineStageFlagBits::eComputeShader;
        deps[1].dstAccessMask = vk::AccessFlagBits::eShaderRead;

        vk::RenderPassCreateInfo ci{};
        ci.attachmentCount = 1; ci.pAttachments  = &da;
        ci.subpassCount    = 1; ci.pSubpasses    = &sp;
        ci.dependencyCount = 2; ci.pDependencies = deps.data();
        m_cascadeRP = m_device.createRenderPass(ci);
    }
}

// ---------------------------------------------------------------------------
inline void Pipeline::createFramebuffers() {
    {
        std::array<vk::ImageView, 3> views{m_gNormal.view, m_velocity.view,
                                           m_prepassDepth.view};
        vk::FramebufferCreateInfo fi{};
        fi.renderPass      = m_prepassRP;
        fi.attachmentCount = 3;
        fi.pAttachments    = views.data();
        fi.width           = m_screenExtent.width;
        fi.height          = m_screenExtent.height;
        fi.layers          = 1;
        m_prepassFB = m_device.createFramebuffer(fi);
    }

    // v40: image cascade baru saja dibuat/dibuat ulang, jadi isi cache tidak
    // lagi mewakili apa pun. Tanpa reset ini, resize jendela akan membuat
    // layer yang di-cache dipakai ulang padahal isinya sudah hilang.
    m_cascadeEverRendered.fill(false);

    for (uint32_t i = 0; i < Cfg::NUM_CASCADES; ++i) {
        vk::FramebufferCreateInfo fi{};
        fi.renderPass      = m_cascadeRP;
        fi.attachmentCount = 1;
        fi.pAttachments    = &m_cascadeLayerViews[i];
        fi.width           = m_cascadeExtent.width;
        fi.height          = m_cascadeExtent.height;
        fi.layers          = 1;
        m_cascadeFB[i] = m_device.createFramebuffer(fi);
    }
}

// ---------------------------------------------------------------------------
inline void Pipeline::createGraphicsPipelines(const InitInfo& info) {
    // Saring atribut vertex per pipeline.
    //
    // glslc -O membuang input yang tidak terpakai dari interface SPIR-V. Jadi
    // meskipun shader MENDEKLARASIKAN `layout(location = 1) in vec3 inColor;`,
    // kalau tidak dipakai ia lenyap dari modul hasil kompilasi, dan validation
    // layer melapor "Vertex attribute at location N not consumed by vertex
    // shader". Solusinya bukan menambah deklarasi di shader, tapi berhenti
    // mengirim atribut yang memang tidak dibutuhkan pipeline itu.
    auto pickAttribs = [&info](std::initializer_list<uint32_t> locations) {
        std::vector<vk::VertexInputAttributeDescription> out;
        for (const auto& a : info.vertexAttributes)
            for (uint32_t loc : locations)
                if (a.location == loc) { out.push_back(a); break; }
        return out;
    };

    // {0, 2, 3}: posisi + normal + posisi frame sebelumnya.
    // Lokasi 3 HANYA diminta di sini. Pass cascade tidak memerlukannya — ia
    // cuma menulis kedalaman dari sudut pandang matahari dan tidak mengenal
    // motion vector — jadi memintanya di sana hanya menambah baca memori.
    const auto prepassAttribs = pickAttribs({0, 2, 3});
    // {0, 2}: normal ikut dikirim karena normal offset kini dikerjakan di sisi
    // caster, di dalam csm_shadow.vert.
    const auto cascadeAttribs = pickAttribs({0, 2});

    vk::PipelineVertexInputStateCreateInfo viPrepass{};
    viPrepass.vertexBindingDescriptionCount   = 1;
    viPrepass.pVertexBindingDescriptions      = &info.vertexBinding;
    viPrepass.vertexAttributeDescriptionCount = static_cast<uint32_t>(prepassAttribs.size());
    viPrepass.pVertexAttributeDescriptions    = prepassAttribs.data();

    vk::PipelineVertexInputStateCreateInfo viCascade{};
    viCascade.vertexBindingDescriptionCount   = 1;
    viCascade.pVertexBindingDescriptions      = &info.vertexBinding;
    viCascade.vertexAttributeDescriptionCount = static_cast<uint32_t>(cascadeAttribs.size());
    viCascade.pVertexAttributeDescriptions    = cascadeAttribs.data();

    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.topology = vk::PrimitiveTopology::eTriangleList;

    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineDepthStencilStateCreateInfo ds{};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = vk::CompareOp::eLess;

    // ---- Pre-pass ----------------------------------------------------------
    {
        auto vm = loadShader("prepass.vert.spv");
        auto fm = loadShader("prepass.frag.spv");

        std::array<vk::PipelineShaderStageCreateInfo, 2> stages{};
        stages[0].stage = vk::ShaderStageFlagBits::eVertex;
        stages[0].module = vm; stages[0].pName = "main";
        stages[1].stage = vk::ShaderStageFlagBits::eFragment;
        stages[1].module = fm; stages[1].pName = "main";

        vk::Viewport vp{0.0f, 0.0f,
                        static_cast<float>(m_screenExtent.width),
                        static_cast<float>(m_screenExtent.height), 0.0f, 1.0f};
        vk::Rect2D   sc{vk::Offset2D{0, 0}, m_screenExtent};
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount = 1; vs.pViewports = &vp;
        vs.scissorCount  = 1; vs.pScissors  = &sc;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode = vk::PolygonMode::eFill;
        rz.lineWidth   = 1.0f;
        rz.cullMode    = vk::CullModeFlagBits::eNone;
        rz.frontFace   = vk::FrontFace::eCounterClockwise;

        std::array<vk::PipelineColorBlendAttachmentState, 2> cba{};
        for (auto& a : cba) {
            a.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                             | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
            a.blendEnable = VK_FALSE;
        }
        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount = 2;
        cb.pAttachments    = cba.data();

        vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex, 0,
                                  static_cast<uint32_t>(sizeof(glm::mat4) * 2)};
        vk::PipelineLayoutCreateInfo pli{};
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        m_prepassLayout = m_device.createPipelineLayout(pli);

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount = 2; pi.pStages = stages.data();
        pi.pVertexInputState   = &viPrepass; pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vs;  pi.pRasterizationState = &rz;
        pi.pMultisampleState   = &ms;  pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.layout     = m_prepassLayout;
        pi.renderPass = m_prepassRP;
        m_prepassPipeline = m_device.createGraphicsPipeline(nullptr, pi).value;

        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);
    }

    // ---- Cascade shadow ----------------------------------------------------
    {
        auto vm = loadShader("csm_shadow.vert.spv");
        auto fm = loadShader("csm_shadow.frag.spv");

        std::array<vk::PipelineShaderStageCreateInfo, 2> stages{};
        stages[0].stage = vk::ShaderStageFlagBits::eVertex;
        stages[0].module = vm; stages[0].pName = "main";
        stages[1].stage = vk::ShaderStageFlagBits::eFragment;
        stages[1].module = fm; stages[1].pName = "main";

        vk::Viewport vp{0.0f, 0.0f,
                        static_cast<float>(m_cascadeExtent.width),
                        static_cast<float>(m_cascadeExtent.height), 0.0f, 1.0f};
        vk::Rect2D   sc{vk::Offset2D{0, 0}, m_cascadeExtent};
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount = 1; vs.pViewports = &vp;
        vs.scissorCount  = 1; vs.pScissors  = &sc;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode = vk::PolygonMode::eFill;
        rz.lineWidth   = 1.0f;
        // eNone: geometri level ini tipis/single-sided. Front-face culling akan
        // membuat dinding hilang dari shadow map.
        rz.cullMode    = vk::CullModeFlagBits::eNone;
        rz.frontFace   = vk::FrontFace::eCounterClockwise;
        rz.depthBiasEnable         = VK_TRUE;
        rz.depthBiasConstantFactor = Cfg::RASTER_DEPTH_BIAS_CONST;
        rz.depthBiasSlopeFactor    = Cfg::RASTER_DEPTH_BIAS_SLOPE;

        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount = 0;

        // 64 -> 80 byte: mat4 cascadeViewProj + vec4 (arah cahaya, ukuran texel).
        vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex, 0,
                                  static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(glm::vec4))};
        vk::PipelineLayoutCreateInfo pli{};
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        m_cascadeLayout = m_device.createPipelineLayout(pli);

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount = 2; pi.pStages = stages.data();
        pi.pVertexInputState   = &viCascade; pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vs;  pi.pRasterizationState = &rz;
        pi.pMultisampleState   = &ms;  pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.layout     = m_cascadeLayout;
        pi.renderPass = m_cascadeRP;
        m_cascadePipeline = m_device.createGraphicsPipeline(nullptr, pi).value;

        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);
    }
}

// ---------------------------------------------------------------------------
inline void Pipeline::createComputePipelines() {
    const auto CS = vk::ShaderStageFlagBits::eCompute;

    auto makeLayout = [&](const std::vector<vk::DescriptorSetLayoutBinding>& b) {
        vk::DescriptorSetLayoutCreateInfo ci{};
        ci.bindingCount = static_cast<uint32_t>(b.size());
        ci.pBindings    = b.data();
        return m_device.createDescriptorSetLayout(ci);
    };
    auto makePL = [&](vk::DescriptorSetLayout dsl) {
        vk::PipelineLayoutCreateInfo ci{};
        ci.setLayoutCount = 1;
        ci.pSetLayouts    = &dsl;
        return m_device.createPipelineLayout(ci);
    };
    auto makePipe = [&](const std::string& file, vk::PipelineLayout pl) {
        auto sm = loadShader(file);
        vk::PipelineShaderStageCreateInfo st{};
        st.stage = CS; st.module = sm; st.pName = "main";
        vk::ComputePipelineCreateInfo ci{};
        ci.stage = st; ci.layout = pl;
        auto p = m_device.createComputePipeline(nullptr, ci).value;
        m_device.destroyShaderModule(sm);
        return p;
    };

    const auto UB  = vk::DescriptorType::eUniformBuffer;
    const auto CIS = vk::DescriptorType::eCombinedImageSampler;
    const auto SI  = vk::DescriptorType::eStorageImage;

    m_resolveDSL = makeLayout({
        {0, UB,  1, CS}, {1, CIS, 1, CS}, {2, CIS, 1, CS},
        {3, CIS, 1, CS}, {4, CIS, 1, CS}, {5, SI,  1, CS},
        {6, CIS, 1, CS}});                       // 6 = LUT blue noise
    // binding 6 = aoRaw: AO ikut diakumulasi temporal bersama bayangan
    // 7 = gNormal (uji disoklusi), 8/9 = momen masuk/keluar (variansi SVGF)
    m_temporalDSL = makeLayout({
        {0, UB,  1, CS}, {1, CIS, 1, CS}, {2, CIS, 1, CS},
        {3, CIS, 1, CS}, {4, CIS, 1, CS}, {5, SI,  1, CS}, {6, CIS, 1, CS},
        {7, CIS, 1, CS}, {8, CIS, 1, CS}, {9, SI,  1, CS}});
    // binding 3 = gNormal: pass blur butuh normal piksel tengah untuk uji bidang
    // (memutus rembesan bayangan dinding ke lantai di pertemuan keduanya).
    m_atrousDSL = makeLayout({
        {0, UB,  1, CS}, {1, CIS, 1, CS}, {2, SI,  1, CS}, {3, CIS, 1, CS},
        {4, CIS, 1, CS}, {5, CIS, 1, CS}});   // 5 = shadowResolve (.g lebar penumbra)

    // AO: UBO + depth + normal + gambar keluaran
    m_aoDSL = makeLayout({
        {0, UB,  1, CS}, {1, CIS, 1, CS}, {2, CIS, 1, CS}, {3, SI, 1, CS},
        {4, CIS, 1, CS}});                       // 4 = LUT blue noise

    // Volumetric: UBO + depth + cascade (sampler compare) + gambar keluaran
    m_volDSL = makeLayout({
        {0, UB,  1, CS}, {1, CIS, 1, CS}, {2, CIS, 1, CS}, {3, SI, 1, CS},
        {4, CIS, 1, CS}});                       // 4 = LUT blue noise

    m_volPL      = makePL(m_volDSL);
    m_aoPL       = makePL(m_aoDSL);
    m_resolvePL  = makePL(m_resolveDSL);
    m_temporalPL = makePL(m_temporalDSL);
    // A-trous dijalankan berkali-kali dengan shader yang SAMA; stride dan
    // penanda iterasi terakhir dikirim lewat push constant.
    {
        vk::PushConstantRange pcr{CS, 0, static_cast<uint32_t>(sizeof(int32_t) * 4)};
        vk::PipelineLayoutCreateInfo ci{};
        ci.setLayoutCount = 1; ci.pSetLayouts = &m_atrousDSL;
        ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr;
        m_atrousPL = m_device.createPipelineLayout(ci);
    }

    m_aoPipe       = makePipe("ao_resolve.comp.spv",     m_aoPL);
    m_volPipe      = makePipe("volumetric.comp.spv",     m_volPL);
    m_resolvePipe  = makePipe("csm_resolve.comp.spv",    m_resolvePL);
    m_temporalPipe = makePipe("shadow_temporal.comp.spv", m_temporalPL);
    m_atrousPipe   = makePipe("svgf_atrous.comp.spv",    m_atrousPL);
}

// ---------------------------------------------------------------------------
inline void Pipeline::createDescriptors() {
    const uint32_t N = m_framesInFlight;
    const uint32_t setsPerSlot = 10;     // ao + vol + resolve + 2 temporal + 2 atrous0 + 3 atrousN

    std::array<vk::DescriptorPoolSize, 3> ps{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer,        N * 12},
        // 54 -> 59: lima set a-trous (2 atrous0 + 3 atrousN) masing-masing
        // mendapat satu combined image sampler baru, yaitu shadowResolve.
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, N * 59},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage,         N * 14}};

    vk::DescriptorPoolCreateInfo dpi{};
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes    = ps.data();
    dpi.maxSets       = N * setsPerSlot;
    m_descPool = m_device.createDescriptorPool(dpi);

    auto alloc = [&](vk::DescriptorSetLayout dsl) {
        vk::DescriptorSetAllocateInfo ai{};
        ai.descriptorPool     = m_descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &dsl;
        return m_device.allocateDescriptorSets(ai)[0];
    };

    m_aoSets.resize(N);
    m_volSets.resize(N);
    m_resolveSets.resize(N);
    m_temporalSets.resize(N);
    m_atrous0Sets.resize(N);
    m_atrousNSets.resize(N);

    const auto GEN = vk::ImageLayout::eGeneral;
    const auto SRO = vk::ImageLayout::eShaderReadOnlyOptimal;

    for (uint32_t s = 0; s < N; ++s) {
        vk::DescriptorBufferInfo ubi{m_uboBuffers[s], 0, sizeof(ShadowUBO)};

        // ---- ao ------------------------------------------------------------
        m_aoSets[s] = alloc(m_aoDSL);
        {
            vk::DescriptorImageInfo iDepth {m_samplerScreen, m_prepassDepth.view, SRO};
            vk::DescriptorImageInfo iNormal{m_samplerScreen, m_gNormal.view,      SRO};
            vk::DescriptorImageInfo iOut   {nullptr,         m_aoRaw.view,        GEN};

            vk::DescriptorImageInfo iBN   {m_samplerBlueNoise, m_blueNoise.view, SRO};
            std::array<vk::WriteDescriptorSet, 5> w{};
            w[0] = {m_aoSets[s], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubi};
            w[1] = {m_aoSets[s], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iDepth};
            w[2] = {m_aoSets[s], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iNormal};
            w[3] = {m_aoSets[s], 3, 0, 1, vk::DescriptorType::eStorageImage, &iOut};
            w[4] = {m_aoSets[s], 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iBN};
            m_device.updateDescriptorSets(5, w.data(), 0, nullptr);
        }

        // ---- volumetric -----------------------------------------------------
        m_volSets[s] = alloc(m_volDSL);
        {
            vk::DescriptorImageInfo iDepth {m_samplerScreen,     m_prepassDepth.view, SRO};
            vk::DescriptorImageInfo iCasCmp{m_samplerCascadeCmp, m_cascades.view,     SRO};
            vk::DescriptorImageInfo iOut   {nullptr,             m_volumetric.view,   GEN};

            vk::DescriptorImageInfo iBN   {m_samplerBlueNoise, m_blueNoise.view, SRO};
            std::array<vk::WriteDescriptorSet, 5> w{};
            w[0] = {m_volSets[s], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubi};
            w[1] = {m_volSets[s], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iDepth};
            w[2] = {m_volSets[s], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iCasCmp};
            w[3] = {m_volSets[s], 3, 0, 1, vk::DescriptorType::eStorageImage, &iOut};
            w[4] = {m_volSets[s], 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iBN};
            m_device.updateDescriptorSets(5, w.data(), 0, nullptr);
        }

        // ---- resolve ------------------------------------------------------
        m_resolveSets[s] = alloc(m_resolveDSL);
        {
            vk::DescriptorImageInfo iDepth {m_samplerScreen,     m_prepassDepth.view, SRO};
            vk::DescriptorImageInfo iNormal{m_samplerScreen,     m_gNormal.view,      SRO};
            vk::DescriptorImageInfo iCas   {m_samplerCascade,    m_cascades.view,     SRO};
            vk::DescriptorImageInfo iCasCmp{m_samplerCascadeCmp, m_cascades.view,     SRO};
            vk::DescriptorImageInfo iOut   {nullptr,             m_shadowResolve.view, GEN};

            std::array<vk::WriteDescriptorSet, 7> w{};
            w[0] = {m_resolveSets[s], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubi};
            w[1] = {m_resolveSets[s], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iDepth};
            w[2] = {m_resolveSets[s], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iNormal};
            w[3] = {m_resolveSets[s], 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iCas};
            w[4] = {m_resolveSets[s], 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iCasCmp};
            w[5] = {m_resolveSets[s], 5, 0, 1, vk::DescriptorType::eStorageImage, &iOut};
            vk::DescriptorImageInfo iBNr{m_samplerBlueNoise, m_blueNoise.view, SRO};
            w[6] = {m_resolveSets[s], 6, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iBNr};
            m_device.updateDescriptorSets(7, w.data(), 0, nullptr);
        }

        // ---- temporal + blurH (bergantung parity ping-pong) ---------------
        for (uint32_t p = 0; p < 2; ++p) {
            const Img& histIn  = m_history[1 - p];
            const Img& histOut = m_history[p];

            m_temporalSets[s][p] = alloc(m_temporalDSL);
            {
                vk::DescriptorImageInfo iRes  {m_samplerScreen, m_shadowResolve.view, GEN};
                vk::DescriptorImageInfo iHist {m_samplerScreen, histIn.view,          GEN};
                vk::DescriptorImageInfo iVel  {m_samplerScreen, m_velocity.view,      SRO};
                vk::DescriptorImageInfo iDepth{m_samplerScreen, m_prepassDepth.view,  SRO};
                vk::DescriptorImageInfo iOut  {nullptr,         histOut.view,         GEN};

                vk::DescriptorImageInfo iAo   {m_samplerScreen, m_aoRaw.view,         GEN};

                std::array<vk::WriteDescriptorSet, 10> w{};
                w[0] = {m_temporalSets[s][p], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubi};
                w[1] = {m_temporalSets[s][p], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iRes};
                w[2] = {m_temporalSets[s][p], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iHist};
                w[3] = {m_temporalSets[s][p], 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iVel};
                w[4] = {m_temporalSets[s][p], 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iDepth};
                w[5] = {m_temporalSets[s][p], 5, 0, 1, vk::DescriptorType::eStorageImage, &iOut};
                w[6] = {m_temporalSets[s][p], 6, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iAo};

                vk::DescriptorImageInfo iNrmT{m_samplerScreen, m_gNormal.view,     SRO};
                vk::DescriptorImageInfo iMomI{m_samplerScreen, m_moments[1 - p].view, GEN};
                vk::DescriptorImageInfo iMomO{nullptr,         m_moments[p].view,  GEN};
                w[7] = {m_temporalSets[s][p], 7, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iNrmT};
                w[8] = {m_temporalSets[s][p], 8, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iMomI};
                w[9] = {m_temporalSets[s][p], 9, 0, 1, vk::DescriptorType::eStorageImage, &iMomO};
                m_device.updateDescriptorSets(10, w.data(), 0, nullptr);
            }

        }

        // ---- a-trous ---------------------------------------------------------
        // Rantainya: history[p] -> svgfA -> svgfB -> svgfA -> shadowFinal.
        // Iterasi 0 masukannya ikut parity history, tiga sisanya tetap.
        auto writeAtrous = [&](vk::DescriptorSet set, const Img& in, const Img& out) {
            vk::DescriptorImageInfo iIn {m_samplerScreen, in.view,        GEN};
            vk::DescriptorImageInfo iOut{nullptr,         out.view,       GEN};
            vk::DescriptorImageInfo iNrm{m_samplerScreen, m_gNormal.view, SRO};
            vk::DescriptorImageInfo iAoR{m_samplerScreen, m_aoRaw.view,   GEN};
            // Lebar penumbra dibaca dari buffer PRA-temporal. Disengaja: nilai
            // itu cuma dipakai untuk menentukan LEBAR filter, bukan sebagai
            // sinyal yang difilter, jadi ia tidak perlu diredam — dan meredamnya
            // justru akan menyeret lebar penumbra melintasi tepi siluet.
            // m_shadowResolve tidak ditimpa siapa pun sesudah csm_resolve, jadi
            // isinya masih valid sepanjang ketiga dispatch a-trous.
            vk::DescriptorImageInfo iRes{m_samplerScreen, m_shadowResolve.view, GEN};
            std::array<vk::WriteDescriptorSet, 6> w{};
            w[0] = {set, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubi};
            w[1] = {set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iIn};
            w[2] = {set, 2, 0, 1, vk::DescriptorType::eStorageImage, &iOut};
            w[3] = {set, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iNrm};
            w[4] = {set, 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iAoR};
            w[5] = {set, 5, 0, 1, vk::DescriptorType::eCombinedImageSampler, &iRes};
            m_device.updateDescriptorSets(6, w.data(), 0, nullptr);
        };

        for (uint32_t p = 0; p < 2; ++p) {
            m_atrous0Sets[s][p] = alloc(m_atrousDSL);
            writeAtrous(m_atrous0Sets[s][p], m_history[p], m_svgfA);
        }
        // Rantai TIGA iterasi: history[p] -> svgfA -> svgfB -> shadowFinal.
        // Iterasi terakhir HARUS menulis shadowFinal, karena itulah yang dibaca
        // main.frag. Set ketiga disiapkan untuk kalau nanti iterasinya ditambah
        // jadi empat; ia tidak dipakai sekarang.
        m_atrousNSets[s][0] = alloc(m_atrousDSL);
        writeAtrous(m_atrousNSets[s][0], m_svgfA, m_svgfB);
        m_atrousNSets[s][1] = alloc(m_atrousDSL);
        writeAtrous(m_atrousNSets[s][1], m_svgfB, m_shadowFinal);
        m_atrousNSets[s][2] = alloc(m_atrousDSL);
        writeAtrous(m_atrousNSets[s][2], m_svgfA, m_shadowFinal);
    }
}

// ---------------------------------------------------------------------------
// Semua image storage dipindahkan sekali ke eGeneral dan DIBIARKAN di sana
// seumur hidup aplikasi. Menghindari layout ping-pong tiap frame: satu-satunya
// sinkronisasi yang dibutuhkan setelah ini adalah memory barrier antar dispatch.
inline void Pipeline::transitionStorageImages(const InitInfo& info) {
    vk::CommandBufferAllocateInfo ai{};
    ai.commandPool        = info.commandPool;
    ai.level              = vk::CommandBufferLevel::ePrimary;
    ai.commandBufferCount = 1;
    vk::CommandBuffer cmd = m_device.allocateCommandBuffers(ai)[0];

    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(bi);

    std::vector<vk::Image> imgs = {
        m_shadowResolve.image, m_history[0].image, m_history[1].image,
        m_svgfA.image, m_shadowFinal.image, m_aoRaw.image, m_volumetric.image,
        m_svgfB.image, m_moments[0].image, m_moments[1].image};

    std::vector<vk::ImageMemoryBarrier> barriers;
    barriers.reserve(imgs.size());
    for (auto img : imgs) {
        vk::ImageMemoryBarrier b{};
        b.oldLayout           = vk::ImageLayout::eUndefined;
        b.newLayout           = vk::ImageLayout::eGeneral;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.srcAccessMask       = {};
        b.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
        barriers.push_back(b);
    }

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eTransfer,
                        {}, 0, nullptr, 0, nullptr,
                        static_cast<uint32_t>(barriers.size()), barriers.data());

    // Nolkan isi semua image. Tanpa ini, history di frame pertama berisi memori
    // sampah; shadow_temporal memang menolaknya lewat uji guideWorld <= 0, tapi
    // sampah bernilai positif bisa lolos dan bikin satu-dua frame pertama kotor.
    vk::ClearColorValue zero{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};
    vk::ImageSubresourceRange full{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    for (auto img : imgs)
        cmd.clearColorImage(img, vk::ImageLayout::eGeneral, &zero, 1, &full);

    vk::MemoryBarrier post{};
    post.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    post.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader,
                        {}, 1, &post, 0, nullptr, 0, nullptr);
    cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    // Overload ArrayProxy — mengembalikan void saat exception aktif, jadi tidak
    // memicu warning [[nodiscard]] seperti versi (count, ptr, fence).
    info.graphicsQueue.submit(si, nullptr);
    info.graphicsQueue.waitIdle();
    m_device.freeCommandBuffers(info.commandPool, 1, &cmd);
}

// ---------------------------------------------------------------------------
// PERHITUNGAN CASCADE
// ---------------------------------------------------------------------------
inline void Pipeline::updateCascades(const FrameInput& in, uint32_t frameSlot) {
    ShadowUBO ubo{};
    SlotData& slot = m_slots[frameSlot];

    const glm::mat4 viewProj = in.proj * in.view;

    slot.viewProj     = viewProj;
    // Aplikasi yang menyediakan prevViewProj sejak TAA aktif, karena hanya ia
    // yang tahu jitter frame ini. Modul ini tidak lagi menebaknya sendiri.
    slot.prevViewProj = in.hasPrevVP ? in.prevViewProj
                                     : (m_hasHistory ? m_lastViewProj : viewProj);
    slot.parity       = m_frameCounter & 1u;

    m_lastViewProj = viewProj;
    m_hasHistory   = true;

    // ---- Practical split scheme (Zhang et al.) -----------------------------
    const float nearZ = in.nearZ;
    // Batas cascade dipangkas ke jarak yang BENAR-BENAR perlu dibayangi.
    //
    // Ini manfaat SDSM tanpa shader dan tanpa readback. SDSM sungguhan
    // mereduksi depth buffer di GPU lalu hasilnya harus kembali ke CPU tempat
    // matriks cascade dibangun — dan readback itu berlatensi dua frame, yaitu
    // 4 DETIK pada 0,5 FPS. Cascade akan terpasang untuk pemandangan yang sudah
    // ditinggalkan. Di sini jaraknya diturunkan dari kotak pembatas geometri
    // yang sudah dihitung CPU saat level dibangun, jadi latensinya nol.
    //
    // Untuk level 22,5 x 15 m ini, 100 m menyusut jadi sekitar 35 m, dan texel
    // tiap cascade mengecil 2,6-3,4x:
    //     C0 0,18 -> 0,07 cm   C3 1,12 -> 0,48 cm
    //     C4 2,58 -> 1,02 cm   C5 6,94 -> 2,39 cm
    // Efeknya paling terasa pada geometri tipis. Ranting terkurus pohon
    // (jari-jari 2,2 cm) tadinya cuma 0,63 texel di C5 alias HILANG, dan
    // 1,71 texel di C4 alias berkedip; sekarang 1,84 dan 4,32 texel.
    const float fitDist = (in.shadowFitDist > 0.0f)
                        ? std::min(in.shadowFitDist, Cfg::SHADOW_MAX_DISTANCE)
                        : Cfg::SHADOW_MAX_DISTANCE;
    const float farZ  = std::min(in.farZ, fitDist);

    float splits[Cfg::NUM_CASCADES + 1];
    splits[0] = nearZ;
    for (uint32_t i = 1; i <= Cfg::NUM_CASCADES; ++i) {
        float si   = static_cast<float>(i) / static_cast<float>(Cfg::NUM_CASCADES);
        float logS = nearZ * std::pow(farZ / nearZ, si);
        float uniS = nearZ + (farZ - nearZ) * si;
        splits[i]  = Cfg::CASCADE_LAMBDA * logS + (1.0f - Cfg::CASCADE_LAMBDA) * uniS;
    }

    const glm::vec3 L = glm::normalize(in.lightDir);
    const glm::vec3 up = (std::fabs(L.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                                  : glm::vec3(0.0f, 1.0f, 0.0f);

    const float tanV = std::tan(in.fovYRadians * 0.5f);
    const float tanH = tanV * in.aspect;

    for (uint32_t c = 0; c < Cfg::NUM_CASCADES; ++c) {
        const float dn = splits[c];
        const float df = splits[c + 1];

        // 8 sudut sub-frustum di world space
        glm::vec3 corners[8];
        int k = 0;
        for (int fi = 0; fi < 2; ++fi) {
            float d      = (fi == 0) ? dn : df;
            glm::vec3 cc = in.cameraPos + in.cameraForward * d;
            glm::vec3 rx = in.cameraRight * (tanH * d);
            glm::vec3 uy = in.cameraUp    * (tanV * d);
            corners[k++] = cc - rx - uy;
            corners[k++] = cc + rx - uy;
            corners[k++] = cc - rx + uy;
            corners[k++] = cc + rx + uy;
        }

        // ---- Bounding SPHERE, bukan AABB ----------------------------------
        // Sphere invarian terhadap rotasi kamera, jadi ukuran cascade tidak
        // berubah saat pemain memutar pandangan. Kalau memakai AABB, extent
        // ortho berubah tiap frame dan bayangan akan berkedip/menggigil.
        glm::vec3 center(0.0f);
        for (auto& p : corners) center += p;
        center /= 8.0f;

        float radius = 0.0f;
        for (auto& p : corners) radius = std::max(radius, glm::length(p - center));

        // Kuantisasi radius: menghilangkan sisa perubahan kecil akibat presisi float
        radius = std::ceil(radius * 16.0f) / 16.0f;

        // ---- Padding untuk snapping kasar (v40) ----------------------------
        // Pusat cascade dikunci ke kisi q texel di bawah. Dengan std::floor,
        // pergeserannya bisa mencapai q texel penuh per sumbu, jadi radiusnya
        // harus diberi ruang sebanyak itu — kalau tidak, bola pembatas frustum
        // bisa keluar dari kotak ortho dan bayangan hilang di pinggir layar.
        // Dipakai (q+1) texel, bukan q: texel akhir sedikit lebih besar dari
        // texel yang dipakai menghitung padding karena radiusnya baru saja
        // tumbuh, dan satu texel cadangan menutup selisih itu dengan aman.
        const uint32_t snapQ = (Cfg::CASCADE_CACHE && c < 6)
                             ? Cfg::CASCADE_CACHE_SNAP_TEXELS[c] : 1u;
        if (snapQ > 1) {
            const float texel0 = (radius * 2.0f) / static_cast<float>(Cfg::CASCADE_RES);
            radius += static_cast<float>(snapQ + 1) * texel0;
        }

        const float extentWorld = radius * 2.0f;
        const float texelWorld  = extentWorld / static_cast<float>(Cfg::CASCADE_RES);

        // ---- PENGUNCIAN PUSAT KE KISI TEXEL --------------------------------
        //
        // Bounding sphere di atas sudah membuat UKURAN cascade tidak berubah
        // saat kamera berputar. Tapi POSISI-nya masih meluncur bebas saat
        // kamera bergeser: pusatnya mengikuti kamera secara kontinu, sehingga
        // kisi texel shadow map ikut meluncur terhadap dunia. Akibatnya tepi
        // bayangan merayap — artefak klasik yang di setiap engine AAA ditutup
        // dengan mengunci pusat ke kelipatan texel.
        //
        // Di engine ini akibatnya lebih dari sekadar kilau. Geseran satu texel
        // mengubah nilai bayangan di penumbra sekitar 0,25 satuan, sementara
        // ambang penolakan riwayat yang baru dipasang cuma 1,5 x 0,09 = 0,135.
        // Artinya TANPA penguncian, setiap langkah kamera akan menolak riwayat
        // di sepanjang tepi bayangan, dan penekanan derau 6,24x itu hilang
        // persis saat pemain bergerak. Penguncian ini melindungi penyaring
        // temporal, bukan cuma memperhalus tepi.
        //
        // Sumbunya harus PERSIS sumbu yang nanti dipakai glm::lookAt, kalau
        // tidak penguncian dilakukan di ruang yang salah dan tidak menempel ke
        // kisi mana pun. lookAt memakai f = normalize(center - eye) = -L,
        // s = normalize(cross(f, up)), u = cross(s, f).
        {
            const glm::vec3 fwd   = -L;
            const glm::vec3 sAxis = glm::normalize(glm::cross(fwd, up));
            const glm::vec3 uAxis = glm::cross(sAxis, fwd);

            const float cs = glm::dot(center, sAxis);
            const float cu = glm::dot(center, uAxis);
            // Kisi snapping = snapQ texel. snapQ = 1 memberi perilaku identik
            // dengan sebelum v40.
            const float grid     = texelWorld * static_cast<float>(snapQ);
            const float snappedS = std::floor(cs / grid) * grid;
            const float snappedU = std::floor(cu / grid) * grid;

            // Pergeseran maksimalnya satu texel per sumbu, dan itu aman: radius
            // sudah dikuantisasi ke kelipatan 1/16 m, yaitu 6,25 cm, sementara
            // texel terbesar di jangkauan 39 m cuma 2,68 cm. Jadi sisa longgar
            // dari pembulatan radius selalu lebih besar daripada geserannya,
            // dan bola pembatas tetap tercakup kotak ortho.
            center += sAxis * (snappedS - cs) + uAxis * (snappedU - cu);
        }

        // Padding proporsional: cascade selebar ratusan meter butuh ruang di
        // belakang yang sepadan, kalau tidak caster tinggi terpotong dan
        // bayangannya hilang mendadak.
        const float zPad       = std::max(Cfg::CASCADE_Z_PADDING,
                                          extentWorld * Cfg::CASCADE_Z_PADDING_FRAC);
        const float depthRange = extentWorld + zPad;

        glm::vec3 eye       = center + L * (radius + zPad);
        glm::mat4 lightView = glm::lookAt(eye, center, up);
        glm::mat4 lightProj = glm::ortho(-radius, radius, -radius, radius,
                                         0.0f, depthRange);
        // Tidak ada proj[1][1] *= -1 di sini: shadow map tidak pernah ditampilkan,
        // dan konvensi NDC->UV di csm_resolve.comp sudah dibuat konsisten dengan ini.

        // ---- Texel snapping ------------------------------------------------
        // Menggeser proyeksi agar origin cascade selalu jatuh tepat di batas
        // texel. Tanpa ini, translasi kamera sub-texel membuat tepi bayangan
        // "merayap" — artefak paling mudah terlihat di bayangan statis.
        {
            glm::mat4 vp    = lightProj * lightView;
            glm::vec4 origin = vp * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            origin *= static_cast<float>(Cfg::CASCADE_RES) * 0.5f;

            glm::vec4 rounded = glm::round(origin);
            glm::vec4 offset  = (rounded - origin) * (2.0f / static_cast<float>(Cfg::CASCADE_RES));

            lightProj[3][0] += offset.x;
            lightProj[3][1] += offset.y;
        }

        glm::mat4 cascadeVP = lightProj * lightView;

        // ---- KEPUTUSAN CACHE (v40) -----------------------------------------
        //
        // Dua syarat, dua-duanya harus terpenuhi untuk memakai ulang layer ini:
        //
        //   [1] cascadeVP keluar BIT-IDENTIK dengan matriks yang dipakai saat
        //       layer ini terakhir dirender. Bukan "mirip" — identik. Kalau
        //       identik, proyeksi tiap verteks statis ke shadow map juga identik,
        //       jadi isi depth buffer-nya pasti sama. Tidak ada yang ditebak.
        //
        //   [2] tidak ada caster bergerak yang menyentuh kotak cascade ini.
        //       Matriks identik tidak menjamin ISI identik kalau ada benda yang
        //       pindah di dalamnya. Ujinya konservatif: bola pembatas kotak
        //       ortho (setengah diagonal radius, radius, depthRange) melawan
        //       AABB dinamis. Kelebihan-tolak aman, kekurangan-tolak tidak.
        bool needsRender = true;
        if (Cfg::CASCADE_CACHE && m_cascadeEverRendered[c]) {
            const bool sameMatrix =
                std::memcmp(&cascadeVP, &m_cachedCascadeVP[c], sizeof(glm::mat4)) == 0;

            // v43: caster yang di bawah ambang TIDAK digambar ke cascade ini
            // (lihat casterVisibleInCascade), jadi ia juga tidak boleh
            // membatalkan cache-nya. Dua aturan ini WAJIB memakai rumus yang
            // sama — kalau berbeda, cascade bisa di-cache padahal bolanya
            // masih digambar ke sana, dan bayangannya membeku.
            const float casterTexels = (in.dynamicRadius > 0.0f)
                                     ? (2.0f * in.dynamicRadius / std::max(texelWorld, 1e-6f))
                                     : 0.0f;
            const bool  casterDrawn  = casterTexels >= Cfg::MIN_CASTER_TEXELS;

            bool dynamicTouches = false;
            if (in.hasDynamic && casterDrawn) {
                const float half = 0.5f * depthRange;
                const float rSph = std::sqrt(radius * radius + radius * radius
                                             + half * half);
                const glm::vec3 boxC = center + L * (radius + zPad - half);
                float d2 = 0.0f;
                for (int a = 0; a < 3; ++a) {
                    const float v = boxC[a];
                    if (v < in.dynamicMin[a]) d2 += (in.dynamicMin[a] - v) * (in.dynamicMin[a] - v);
                    else if (v > in.dynamicMax[a]) d2 += (v - in.dynamicMax[a]) * (v - in.dynamicMax[a]);
                }
                dynamicTouches = d2 <= rSph * rSph;
            }
            needsRender = !sameMatrix || dynamicTouches;
        }
        m_lastTexelWorld[c]        = texelWorld;
        slot.cascadeNeedsRender[c] = needsRender;
        if (needsRender) {
            m_cachedCascadeVP[c]     = cascadeVP;
            m_cascadeEverRendered[c] = true;
            ++m_cascadeRenderCount;
        } else {
            ++m_cascadeSkipCount;
            ++m_cascadeSkipPer[c];
        }

        const uint32_t hi = c >> 2, lo = c & 3u;
        slot.cascadeVP[c]            = cascadeVP;
        slot.texelWorld[c]           = texelWorld;
        ubo.cascadeVP[c]             = cascadeVP;
        ubo.cascadeSplitView[hi][lo] = df;
        ubo.cascadeExtentWorld[hi][lo]= extentWorld;
        ubo.cascadeDepthRange[hi][lo]= depthRange;
        ubo.cascadeTexelWorld[hi][lo]= texelWorld;
    }

    // v40: laporan hit-rate cache. Tanpa angka ini caching adalah fitur yang
    // tidak bisa dibuktikan bekerja — dan fitur performa yang tidak terukur
    // adalah fitur yang tidak ada.
    if (Cfg::CASCADE_CACHE) {
        const uint64_t total = m_cascadeRenderCount + m_cascadeSkipCount;
        // v43: 600 keputusan = 100 frame = 200 DETIK di 0,5 FPS. Sesi
        // pengujian user yang 35 detik tidak pernah mencapainya, jadi
        // laporannya tidak pernah tercetak dan caching v40 terlihat seperti
        // omong kosong. 60 keputusan = 10 frame = 20 detik.
        if (total >= 60 && (total % 60) < Cfg::NUM_CASCADES) {
            const double hit = 100.0 * static_cast<double>(m_cascadeSkipCount)
                             / static_cast<double>(total);
            std::string per;
            for (uint32_t k = 0; k < Cfg::NUM_CASCADES; ++k)
                per += " C" + std::to_string(k) + ":" + std::to_string(m_cascadeSkipPer[k]);
            // v49: hit rate saja tidak membuktikan apa-apa soal biaya. Yang
            // dihemat adalah CLEAR + RASTER satu layer 2048x2048 penuh, jadi
            // angka yang berarti adalah texel. 6 layer x 2048^2 = 25,17 juta
            // texel per frame kalau tidak ada yang dilewati.
            const double texPerLayer = double(Cfg::CASCADE_RES) * double(Cfg::CASCADE_RES);
            const double savedM = double(m_cascadeSkipCount) * texPerLayer / 1.0e6;
            const double frames = double(total) / double(Cfg::NUM_CASCADES);
            const double perFrameM = frames > 0.0 ? savedM / frames : 0.0;
            const double fullM  = double(Cfg::NUM_CASCADES) * texPerLayer / 1.0e6;
            LOG_INFO("Shadow", "cache cascade " + fmt1(static_cast<float>(hit))
                     + "% hit (" + std::to_string(m_cascadeSkipCount) + "/"
                     + std::to_string(total) + ") | dilewati per cascade:" + per
                     + " | texel dihemat " + fmt1(static_cast<float>(perFrameM))
                     + " dari " + fmt1(static_cast<float>(fullM)) + " juta per frame");
        }
    }

    ubo.viewProj     = viewProj;              // dipakai ray march contact shadow
    slot.lightDir    = L;
    ubo.invViewProj  = glm::inverse(viewProj);
    ubo.prevViewProj = slot.prevViewProj;
    ubo.view         = in.view;
    ubo.lightDirWorld= glm::vec4(L, 0.0f);
    ubo.cameraPos    = glm::vec4(in.cameraPos, tanV);
    ubo.resolution   = glm::vec4(
        static_cast<float>(m_screenExtent.width),
        static_cast<float>(m_screenExtent.height),
        1.0f / static_cast<float>(m_screenExtent.width),
        1.0f / static_cast<float>(m_screenExtent.height));

    ubo.params0 = glm::vec4(Cfg::SUN_ANGULAR_TAN, Cfg::MIN_PENUMBRA_WORLD,
                            Cfg::MAX_PENUMBRA_WORLD, Cfg::NORMAL_OFFSET_PX);
    ubo.params1 = glm::vec4(Cfg::DEPTH_BIAS_PX, Cfg::TEMPORAL_TAU_MOVING,
                            Cfg::DISOCCLUSION_REL, Cfg::BLUR_MAX_RADIUS_PX);
    ubo.params2 = glm::vec4(static_cast<float>(m_frameCounter), nearZ, in.farZ,
                            Cfg::CASCADE_BLEND_FRAC);
    ubo.params3 = glm::vec4(static_cast<float>(Cfg::CASCADE_RES),
                            1.0f / static_cast<float>(Cfg::CASCADE_RES),
                            Cfg::BLOCKER_SEARCH_WORLD, Cfg::TEMPORAL_TAU_STATIC);
    // Batas atas 0,2 detik MELUMPUHKAN seluruh logika berbasis waktu di hilir.
    //
    // Di 0,5 FPS satu frame nyatanya 2 detik, tapi shader diberi tahu 0,2 —
    // meleset sepuluh kali. Akibatnya exp(-dt/tau) menghitung peredaman untuk
    // frame yang jauh lebih pendek daripada kenyataan, dan riwayat bertahan
    // beberapa detik alih-alih setengah detik. Itu ghosting yang tersisa.
    //
    // Batasnya dinaikkan ke 4 detik: cukup untuk menampung frame paling lambat
    // di rasterizer perangkat lunak, tetap menjaga dari lonjakan liar saat
    // jendela di-resize atau aplikasi baru bangun dari jeda.
    ubo.params4 = glm::vec4(glm::clamp(in.deltaTime, 1.0f / 240.0f, 4.0f),
                            Cfg::MIN_BLUR_RADIUS_PX, Cfg::MIN_PCF_SCREEN_PX,
                            m_cmpLinearOK ? 0.0f : 1.0f);   // 1 = pakai bilinear manual
    // Kalau penyaringan dimatikan, params5.x diisi 3 — nilai yang sudah dipakai
    // temporal dan kedua blur sebagai perintah "teruskan apa adanya". Jalur itu
    // persis DEBUG_MODE 3, yang sudah terbukti bersih di layar. DEBUG_MODE yang
    // disetel manual tetap menang supaya mode diagnostik masih bisa dipakai.
    const float debugOrBypass =
        (Cfg::DEBUG_MODE != 0) ? static_cast<float>(Cfg::DEBUG_MODE)
                               : (Cfg::SHADOW_FILTER_ENABLED ? 0.0f : 3.0f);

    ubo.params5 = glm::vec4(debugOrBypass,
                            farZ,                       // jarak bayangan efektif
                            Cfg::SHADOW_FADE_FRAC,
                            Cfg::VOL_DENSITY);
    // Parameter volumetric SENGAJA menumpang di slot .w yang selama ini kosong
    // (params5, params8, params9), bukan lewat vec4 baru. Menambah field ke
    // ShadowUBO berarti menyunting blok yang sama di ENAM shader sekaligus —
    // dan justru pekerjaan itu yang sudah dua kali menghasilkan build gagal.
    ubo.params8 = glm::vec4(Cfg::AO_MIN_VISIBILITY, Cfg::AO_BLUR_RADIUS_PX,
                            Cfg::AO_MAX_RADIUS_PX, static_cast<float>(Cfg::VOL_STEPS));
    ubo.params9 = glm::vec4(Cfg::NORMAL_OFFSET_MAX_WORLD, Cfg::DEPTH_BIAS_MAX_WORLD,
                            Cfg::MIN_BLUR_RADIUS_PX_SLOW, Cfg::VOL_MAX_DIST);
    ubo.params7 = glm::vec4(Cfg::AO_RADIUS_WORLD,
                            static_cast<float>(Cfg::AO_SLICES),
                            static_cast<float>(Cfg::AO_STEPS),
                            Cfg::AO_INTENSITY);
    ubo.params6 = glm::vec4(Cfg::CONTACT_LENGTH_WORLD,
                            static_cast<float>(Cfg::CONTACT_STEPS),
                            Cfg::CONTACT_THICKNESS,
                            Cfg::CONTACT_FADE_DIST);

    // Statistik untuk laporan sekali di log (lihat cascadeReport()).
    for (uint32_t c = 0; c < Cfg::NUM_CASCADES; ++c) {
        m_statSplit[c]  = ubo.cascadeSplitView[c >> 2][c & 3u];
        m_statExtent[c] = ubo.cascadeExtentWorld[c >> 2][c & 3u];
        m_statTexel[c]  = ubo.cascadeTexelWorld[c >> 2][c & 3u];
    }
    m_statTanHalfFovY = tanV;
    m_statResY        = static_cast<float>(m_screenExtent.height);
    m_hasStats        = true;

    std::memcpy(m_uboMapped[frameSlot], &ubo, sizeof(ShadowUBO));
    ++m_frameCounter;
}

// ---------------------------------------------------------------------------
// PEREKAMAN COMMAND BUFFER
// ---------------------------------------------------------------------------
inline void Pipeline::record(vk::CommandBuffer cmd, uint32_t frameSlot,
                             const DrawSceneFn& drawScene, GpuProfiler* prof) {
    const SlotData& slot = m_slots[frameSlot];
    const uint32_t  p    = slot.parity;

    // ===== PASS 0: Depth pre-pass (depth + normal + velocity) ==============
    {
        std::array<vk::ClearValue, 3> clears{};
        clears[0].color        = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
        clears[1].color        = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
        clears[2].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        vk::RenderPassBeginInfo rp{};
        rp.renderPass        = m_prepassRP;
        rp.framebuffer       = m_prepassFB;
        rp.renderArea.offset = vk::Offset2D{0, 0};
        rp.renderArea.extent = m_screenExtent;
        rp.clearValueCount   = 3;
        rp.pClearValues      = clears.data();

        cmd.beginRenderPass(rp, vk::SubpassContents::eInline);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_prepassPipeline);

        struct { glm::mat4 vp; glm::mat4 prevVP; } pc{slot.viewProj, slot.prevViewProj};
        cmd.pushConstants(m_prepassLayout, vk::ShaderStageFlagBits::eVertex,
                          0, sizeof(pc), &pc);
        drawScene(cmd, -1);
        cmd.endRenderPass();
    }
    if (prof) prof->stamp(cmd, frameSlot, GpuProfiler::Prepass);

    // ===== PASS 1: Render 4 cascade ========================================
    for (uint32_t c = 0; c < Cfg::NUM_CASCADES; ++c) {
        // v40: layer yang di-cache dilewati SELURUHNYA — tidak ada
        // beginRenderPass, jadi tidak ada clear dan tidak ada draw. Isinya
        // tetap seperti frame terakhir yang merendernya, dan layout-nya juga
        // tetap finalLayout dari pass itu, jadi tetap sah untuk disampel.
        if (!slot.cascadeNeedsRender[c]) continue;

        vk::ClearValue clear{};
        clear.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        vk::RenderPassBeginInfo rp{};
        rp.renderPass        = m_cascadeRP;
        rp.framebuffer       = m_cascadeFB[c];
        rp.renderArea.offset = vk::Offset2D{0, 0};
        rp.renderArea.extent = m_cascadeExtent;
        rp.clearValueCount   = 1;
        rp.pClearValues      = &clear;

        cmd.beginRenderPass(rp, vk::SubpassContents::eInline);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_cascadePipeline);
        struct { glm::mat4 vp; glm::vec4 lightAndScale; } cpc{
            // v47: texel dikirim SUDAH diskalakan. Shader tidak perlu tahu
            // konstantanya, dan tidak ada dua tempat yang bisa meleset.
            slot.cascadeVP[c],
            glm::vec4(slot.lightDir, slot.texelWorld[c] * Cfg::CASTER_OFFSET_SCALE)};
        cmd.pushConstants(m_cascadeLayout, vk::ShaderStageFlagBits::eVertex,
                          0, sizeof(cpc), &cpc);
        drawScene(cmd, static_cast<int>(c));
        cmd.endRenderPass();
    }
    if (prof) prof->stamp(cmd, frameSlot, GpuProfiler::Cascades);

    // ===== PASS 2..5: rantai compute =======================================
    const uint32_t gx = (m_screenExtent.width  + 7) / 8;
    const uint32_t gy = (m_screenExtent.height + 7) / 8;

    // -- Ambient occlusion (GTAO) --
    // Dijalankan sebelum resolve; keduanya cuma butuh depth + normal dari
    // pre-pass, jadi tidak saling bergantung. Barrier di bawah menjamin AO
    // sudah selesai sebelum temporal membacanya.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_aoPipe);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_aoPL,
                           0, 1, &m_aoSets[frameSlot], 0, nullptr);
    cmd.dispatch(gx, gy, 1);

    // -- CSM resolve (PCSS) --
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_resolvePipe);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_resolvePL,
                           0, 1, &m_resolveSets[frameSlot], 0, nullptr);
    cmd.dispatch(gx, gy, 1);
    if (prof) prof->stamp(cmd, frameSlot, GpuProfiler::Resolve);
    computeBarrier(cmd, vk::PipelineStageFlagBits::eComputeShader,
                   vk::AccessFlagBits::eShaderRead);

    // -- Temporal reprojection (ping-pong) --
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_temporalPipe);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_temporalPL,
                           0, 1, &m_temporalSets[frameSlot][p], 0, nullptr);
    cmd.dispatch(gx, gy, 1);
    if (prof) prof->stamp(cmd, frameSlot, GpuProfiler::Temporal);
    computeBarrier(cmd, vk::PipelineStageFlagBits::eComputeShader,
                   vk::AccessFlagBits::eShaderRead);

    // -- Volumetric light shaft (setengah resolusi) --
    // Tidak bergantung pada rantai bayangan layar; ia cuma butuh cascade dan
    // depth pre-pass, keduanya sudah siap sejak awal blok compute ini.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_volPipe);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_volPL,
                           0, 1, &m_volSets[frameSlot], 0, nullptr);
    cmd.dispatch((m_volExtent.width + 7) / 8, (m_volExtent.height + 7) / 8, 1);

    // -- A-TROUS SVGF: empat iterasi, satu pipeline ------------------------
    //
    // Menggantikan blur H dan V sepenuhnya. Stride berlipat dua tiap iterasi
    // (1, 2, 4, 8), jadi jangkauan efektifnya 33x33 piksel dengan 4 x 25 tap —
    // bukan 1089 tap yang dibutuhkan filter kotak selebar itu.
    //
    // Rantai gambar: history[p] -> svgfA -> svgfB -> svgfA -> shadowFinal.
    // Iterasi terakhir menulis format yang dibaca main.frag.
    {
        // DUA iterasi, bukan tiga.
        //
        // Jangkauan tiga iterasi (stride 1,2,4) = 29x29 piksel. Itu ukuran
        // denoiser untuk sinyal ray-traced yang mentahnya sangat berisik.
        // Bayangan kita sudah difilter PCSS 12 tap SEBELUM masuk sini, jadi
        // yang tersisa cuma dither sub-texel beramplitudo kecil — dan
        // menyapunya dengan jangkauan 29 piksel ikut meratakan penumbra yang
        // sudah benar. Contact hardening PCSS jadi hilang: tepi yang seharusnya
        // tajam di titik sentuh ikut melebar.
        //
        // Dua iterasi (stride 1,2) = 13x13. Cukup untuk dither, tidak cukup
        // untuk merusak bentuk penumbra.
        // TIGA iterasi: stride 1, 2, 4 -> jangkauan 29 piksel.
        //
        // Dua iterasi sudah cukup selama tidak ada yang bergerak. Begitu bola
        // dinamis masuk, muncul rezim kedua: di pita yang disapu bayangannya,
        // penolakan riwayat membuat histLen kembali 1 tiap frame, jadi
        // penekanan derau temporal di situ NOL dan derau PCF penuh 0,09 sampai
        // apa adanya ke penyaring spasial. Di situlah iterasi ketiga membayar.
        //
        // Diukur pada ramp penumbra sintetis, 30 percobaan per titik:
        //
        //   rezim          iterasi     setia lebar   penekanan derau
        //   pita bola      1,2            0,75           2,12x
        //   pita bola      1,2,4          0,93           2,32x
        //   daerah mapan   1,2            0,69           2,26x
        //   daerah mapan   1,2,4          0,69           2,40x
        //
        // Menang di kedua rezim, dan di pita bola kesetiaan lebar penumbranya
        // justru NAIK — stride lebar membuat taksiran kemiringan lokal pada
        // wl terkoreksi gradien jadi lebih akurat, sehingga ramp dimuluskan
        // sepanjang arahnya sendiri alih-alih dipotong.
        //
        // Uji keamanan yang menahan light bleeding tetap lolos:
        //   tepi keras 1,05 -> 1,27 px, kebocoran sisi gelap +3%,
        //   dasar pita tipis 5 px +2%. Semuanya dapat diabaikan.
        //
        // Iterasi keempat (stride 8) TIDAK ditambahkan: jangkauannya 61 piksel,
        // jauh melewati lebar penumbra mana pun di scene ini.
        const vk::DescriptorSet chain[3] = {
            m_atrous0Sets[frameSlot][p],   // history[p] -> svgfA       (stride 1)
            m_atrousNSets[frameSlot][0],   // svgfA      -> svgfB       (stride 2)
            m_atrousNSets[frameSlot][1],   // svgfB      -> shadowFinal (stride 4)
        };
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_atrousPipe);
        for (int it = 0; it < 3; ++it) {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_atrousPL,
                                   0, 1, &chain[it], 0, nullptr);
            // cfg.y = 1 hanya di iterasi TERAKHIR: di situ bent normal ikut
            // ditulis dan kanal depth tidak lagi diteruskan.
            int32_t pc4[4] = {1 << it, (it == 2) ? 1 : 0, 0, 0};
            cmd.pushConstants(m_atrousPL, vk::ShaderStageFlagBits::eCompute,
                              0, sizeof(pc4), pc4);
            cmd.dispatch(gx, gy, 1);
            if (it < 2)
                computeBarrier(cmd, vk::PipelineStageFlagBits::eComputeShader,
                               vk::AccessFlagBits::eShaderRead);
        }
        if (prof) prof->stamp(cmd, frameSlot, GpuProfiler::BlurV);
    }

    // Serahkan shadowFinal ke fragment shader main pass
    computeBarrier(cmd, vk::PipelineStageFlagBits::eFragmentShader,
                   vk::AccessFlagBits::eShaderRead);
}

// ---------------------------------------------------------------------------
inline void Pipeline::destroy() {
    if (!m_device) return;

    for (uint32_t i = 0; i < m_uboBuffers.size(); ++i) {
        if (m_uboMapped[i]) m_device.unmapMemory(m_uboMemory[i]);
        if (m_uboBuffers[i]) m_device.destroyBuffer(m_uboBuffers[i]);
        if (m_uboMemory[i])  m_device.freeMemory(m_uboMemory[i]);
    }
    m_uboBuffers.clear(); m_uboMemory.clear(); m_uboMapped.clear();

    if (m_descPool) m_device.destroyDescriptorPool(m_descPool);

    for (auto p : {m_resolvePipe, m_temporalPipe, m_atrousPipe, m_aoPipe, m_volPipe})
        if (p) m_device.destroyPipeline(p);
    for (auto l : {m_resolvePL, m_temporalPL, m_atrousPL, m_aoPL, m_volPL})
        if (l) m_device.destroyPipelineLayout(l);
    for (auto d : {m_resolveDSL, m_temporalDSL, m_atrousDSL, m_aoDSL, m_volDSL})
        if (d) m_device.destroyDescriptorSetLayout(d);

    if (m_prepassPipeline) m_device.destroyPipeline(m_prepassPipeline);
    if (m_cascadePipeline) m_device.destroyPipeline(m_cascadePipeline);
    if (m_prepassLayout)   m_device.destroyPipelineLayout(m_prepassLayout);
    if (m_cascadeLayout)   m_device.destroyPipelineLayout(m_cascadeLayout);

    if (m_prepassFB) m_device.destroyFramebuffer(m_prepassFB);
    for (auto fb : m_cascadeFB) if (fb) m_device.destroyFramebuffer(fb);
    if (m_prepassRP) m_device.destroyRenderPass(m_prepassRP);
    if (m_cascadeRP) m_device.destroyRenderPass(m_cascadeRP);

    for (auto s : {m_samplerScreen, m_samplerCascade, m_samplerCascadeCmp})
        if (s) m_device.destroySampler(s);

    for (auto v : m_cascadeLayerViews) if (v) m_device.destroyImageView(v);

    destroyImage(m_prepassDepth); destroyImage(m_gNormal);   destroyImage(m_velocity);
    destroyImage(m_cascades);     destroyImage(m_shadowResolve);
    destroyImage(m_history[0]);   destroyImage(m_history[1]);
    destroyImage(m_svgfA);        destroyImage(m_shadowFinal);
    destroyImage(m_svgfB);        destroyImage(m_moments[0]);
    destroyImage(m_moments[1]);
    destroyImage(m_aoRaw);        destroyImage(m_volumetric);
    destroyImage(m_blueNoise);
    if (m_samplerBlueNoise) m_device.destroySampler(m_samplerBlueNoise);

    m_device = nullptr;
}

} // namespace ShadowAAA

// =============================================================================
// LEVEL MAP
// =============================================================================
std::vector<std::vector<int>> levelMap = {
    {1,1,1,1,1,1,1,1,1,1},{1,0,0,0,0,0,0,0,0,1},{1,0,1,1,0,0,0,1,0,1},{1,0,1,0,0,0,0,0,0,1},
    {1,0,1,0,0,0,0,0,0,1},{1,0,1,0,0,0,0,0,0,1},{1,0,0,0,0,0,0,0,0,1},{1,0,0,0,0,0,0,0,0,1},
    {1,0,0,0,1,1,0,0,0,1},{1,0,0,0,0,0,0,0,0,1},{1,0,0,0,0,0,0,0,0,1},{1,0,0,0,0,0,0,0,0,1},
    {1,0,1,0,0,0,0,1,0,1},{1,0,0,0,0,0,0,0,0,1},{1,1,1,1,1,1,1,1,1,1}
};

// =============================================================================
// DETEKSI SPIR-V BASI (v45)
//
//   Kelas kegagalan yang baru saja memakan dua build: shaders.cpp diedit, C++
//   dikompilasi ulang, tapi ./compile_shaders.sh tidak dijalankan. Binari
//   shader lama tetap dipakai. TIDAK ADA error, tidak ada peringatan, tidak
//   ada gejala — fitur barunya cuma tidak muncul, dan yang tersisa adalah
//   menebak apakah kodenya salah atau build-nya yang tidak sampai.
//
//   Ini menutup celah itu untuk selamanya. Setiap *.spv di folder kerja
//   dibandingkan waktu ubahnya dengan shaders.cpp. Yang lebih tua = basi.
//
//   Sengaja LOG_ERROR, bukan LOG_WARN: konsekuensinya adalah menjalankan
//   engine yang bukan engine yang baru saja kamu tulis.
// =============================================================================
static void checkStaleShaderBinaries(const char* srcName = "shaders.cpp") {
    struct stat srcSt{};
    if (stat(srcName, &srcSt) != 0) return;      // sumber tidak ada, lewati

    DIR* d = opendir(".");
    if (!d) return;

    std::vector<std::string> stale;
    int checked = 0;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 5 || n.compare(n.size() - 4, 4, ".spv") != 0) continue;
        struct stat sp{};
        if (stat(n.c_str(), &sp) != 0) continue;
        ++checked;
        if (sp.st_mtime < srcSt.st_mtime) stale.push_back(n);
    }
    closedir(d);

    if (stale.empty()) {
        LOG_INFO("Shader", std::to_string(checked)
                 + " berkas .spv diperiksa, semuanya lebih baru dari shaders.cpp");
        return;
    }
    std::string list;
    for (size_t i = 0; i < stale.size(); ++i) {
        if (i) list += ", ";
        list += stale[i];
    }
    LOG_ERROR("Shader", "SPIR-V BASI — " + std::to_string(stale.size()) + " dari "
              + std::to_string(checked) + " lebih tua daripada shaders.cpp: " + list);
    LOG_ERROR("Shader", "Yang berjalan adalah shader LAMA. Jalankan "
                        "./compile_shaders.sh lalu ulangi.");
}

// =============================================================================
// PEMBESARAN PETA (v41)
//
// Mengganti levelMap dengan peta MAP_BIG_W x MAP_BIG_H yang memuat peta lama
// utuh di tengahnya. Dipanggil SEKALI sebelum geometri dibangun.
//
// Tiga hal yang dijaga:
//
//  [1] Offset penyalinan bilangan bulat. Generator memusatkan peta di origin,
//      jadi peta lama tetap di koordinat dunia yang sama HANYA kalau selisih
//      ukurannya genap. Di-assert, bukan diasumsikan.
//
//  [2] Dinding tepi peta lama dilubangi di tengah tiap sisi. Tanpa itu pemain
//      terkurung di arena lama dan seluruh kota baru tidak pernah terlihat.
//
//  [3] Generator deterministik (LCG berbiji tetap). Peta yang berubah tiap run
//      akan membatalkan semua pengukuran sebelum-sesudah — pelajaran yang sama
//      dengan SUN_FREEZE.
// =============================================================================
static void enlargeLevelMap() {
    if (!Config::MAP_BIG) return;

    const int oldH = static_cast<int>(levelMap.size());
    const int oldW = static_cast<int>(levelMap[0].size());
    const int W    = Config::MAP_BIG_W;
    const int H    = Config::MAP_BIG_H;
    if (W <= oldW || H <= oldH) return;

    // [1] paritas — kalau meleset, seluruh scene bergeser setengah sel
    if (((W - oldW) & 1) || ((H - oldH) & 1)) {
        LOG_ERROR("Map", "MAP_BIG_W/H harus separitas dengan peta lama; "
                         "pembesaran dibatalkan agar scene tidak bergeser.");
        return;
    }
    const int offX = (W - oldW) / 2;
    const int offZ = (H - oldH) / 2;

    std::vector<std::vector<int>> big(static_cast<size_t>(H),
                                      std::vector<int>(static_cast<size_t>(W), 0));

    // [3] LCG deterministik
    uint32_t rng = Config::MAP_SEED;
    auto next = [&rng]() { rng = rng * 1664525u + 1013904223u; return rng >> 16; };

    const float cx = W * 0.5f, cz = H * 0.5f;
    const float maxr = std::min(cx, cz);
    auto radial = [&](int x, int z) {
        return std::max(std::fabs(x - cx), std::fabs(z - cz)) / maxr;
    };

    // ---- [A] INTI KOTA: padat, rendah, bisa dijelajahi --------------------
    for (int z = 1; z < H - 1; ++z) {
        for (int x = 1; x < W - 1; ++x) {
            if (radial(x, z) > Config::MAP_CORE_R) continue;
            if (x % Config::MAP_BLOCK == 0 || z % Config::MAP_BLOCK == 0) continue;
            if (next() % 100u < static_cast<uint32_t>(Config::MAP_CORE_SKIP)) continue;
            big[static_cast<size_t>(z)][static_cast<size_t>(x)] =
                1 + static_cast<int>(next() % 3u);
        }
    }

    // ---- [B] CINCIN MENENGAH: mengisi jarak C3-C4 -------------------------
    for (int z = 1; z < H - 1; ++z) {
        for (int x = 1; x < W - 1; ++x) {
            if (big[static_cast<size_t>(z)][static_cast<size_t>(x)]) continue;
            const float rr = radial(x, z);
            if (rr <= Config::MAP_CORE_R || rr > Config::MAP_MID_R) continue;
            if (x % Config::MAP_BLOCK == 0 || z % Config::MAP_BLOCK == 0) continue;
            if (next() % 100u < static_cast<uint32_t>(Config::MAP_MID_SKIP)) continue;
            big[static_cast<size_t>(z)][static_cast<size_t>(x)] =
                2 + static_cast<int>(next() % 4u);
        }
    }

    // ---- [C] MENARA LUAR: sedikit tapi BESAR dan TINGGI -------------------
    //
    // Inilah yang membuat cascade terjauh punya sesuatu untuk dibayangi.
    // Di jarak 145 m, kotak 6 m yang dipakai v41 hanya setinggi 2,4 derajat di
    // layar — praktis tidak terlihat, apalagi bayangannya. Menara 32 m di
    // jarak yang sama setinggi 12,3 derajat dan membuang bayangan sepanjang
    // 52 m: itu baru terbaca sebagai bayangan cascade jauh.
    //
    // Sengaja JARANG (MAP_TOWER_SKIP per seribu percobaan ditolak). Menara
    // yang rapat cuma jadi karpet mahal; yang dicari landmark.
    int towers = 0;
    for (int t = 0; t < Config::MAP_TOWER_TRIES; ++t) {
        const int x = static_cast<int>(next() % static_cast<uint32_t>(W));
        const int z = static_cast<int>(next() % static_cast<uint32_t>(H));
        const float rr = radial(x, z);
        if (rr < 0.26f || rr > 0.92f) continue;
        if (next() % 1000u < static_cast<uint32_t>(Config::MAP_TOWER_SKIP)) continue;

        const int tw = 3 + static_cast<int>(next() % 4u);
        const int td = 3 + static_cast<int>(next() % 4u);
        // Makin jauh dari pusat, makin tinggi — siluet kota yang meninggi
        // ke arah cakrawala, dan casternya besar persis di jarak C5.
        const int lvl = std::min(6 + static_cast<int>(rr * 14.0f)
                                   + static_cast<int>(next() % 4u),
                                 Config::MAP_MAX_LEVELS);
        bool placed = false;
        for (int dz = 0; dz < td; ++dz)
            for (int dx = 0; dx < tw; ++dx) {
                const int gx = x + dx, gz = z + dz;
                if (gx <= 0 || gx >= W - 1 || gz <= 0 || gz >= H - 1) continue;
                if (big[static_cast<size_t>(gz)][static_cast<size_t>(gx)]) continue;
                big[static_cast<size_t>(gz)][static_cast<size_t>(gx)] = lvl;
                placed = true;
            }
        if (placed) ++towers;
    }

    // Peta lama disalin UTUH, menimpa apa pun yang digenerate di area itu.
    for (int z = 0; z < oldH; ++z)
        for (int x = 0; x < oldW; ++x)
            big[static_cast<size_t>(z + offZ)][static_cast<size_t>(x + offX)] =
                levelMap[static_cast<size_t>(z)][static_cast<size_t>(x)];

    // Ruang bebas satu sel mengelilingi arena lama, supaya bangunan generate
    // tidak menempel di dindingnya dan pintu keluar tidak langsung buntu.
    for (int z = -1; z <= oldH; ++z)
        for (int x = -1; x <= oldW; ++x) {
            if (z >= 0 && z < oldH && x >= 0 && x < oldW) continue;
            const int gz = z + offZ, gx = x + offX;
            if (gz <= 0 || gz >= H - 1 || gx <= 0 || gx >= W - 1) continue;
            big[static_cast<size_t>(gz)][static_cast<size_t>(gx)] = 0;
        }

    // [2] pintu keluar di tengah tiap sisi arena lama
    const int midX = offX + oldW / 2, midZ = offZ + oldH / 2;
    for (int d = -1; d <= 1; ++d) {
        big[static_cast<size_t>(offZ)][static_cast<size_t>(midX + d)]          = 0;
        big[static_cast<size_t>(offZ + oldH - 1)][static_cast<size_t>(midX + d)] = 0;
        big[static_cast<size_t>(midZ + d)][static_cast<size_t>(offX)]          = 0;
        big[static_cast<size_t>(midZ + d)][static_cast<size_t>(offX + oldW - 1)] = 0;
    }

    levelMap.swap(big);

    int solid = 0, tallest = 0;
    for (auto& row : levelMap)
        for (int v : row) { if (v) ++solid; tallest = std::max(tallest, v); }

    // Angka ini yang menentukan apakah peta masih bisa dijalankan: pre-pass
    // menggambar semuanya tiap frame tanpa culling kamera.
    LOG_INFO("Map", "peta " + std::to_string(W) + "x" + std::to_string(H) + " sel = "
             + std::to_string(static_cast<int>(W * Config::MAP_SCALE)) + " x "
             + std::to_string(static_cast<int>(H * Config::MAP_SCALE)) + " m | "
             + std::to_string(solid) + " sel terisi (anggaran simulasi 2478) | "
             + std::to_string(towers) + " menara | tertinggi "
             + std::to_string(static_cast<int>(tallest * Config::WALL_HEIGHT))
             + " m | arena lama di offset (" + std::to_string(offX) + ","
             + std::to_string(offZ) + ")");
}

// =============================================================================
// GEOMETRY GENERATION — delegasi ke GeometryGenerator class
// Fungsi free ini untuk backward compat dengan createLevelBuffers()
// =============================================================================
void generateLevelGeometry(
    std::vector<Vertex>& outVertices,
    std::vector<uint32_t>& outIndices,
    std::vector<TileInfo>& outTiles)
{
    GeometryGenerator::generateLevelGeometry(outVertices, outIndices, &outTiles);
}

// =============================================================================
// HELPER — readFile delegasi ke ShaderModule::ReadFile dengan throw fallback
// =============================================================================
static std::vector<char> readFile(const std::string& filename) {
    auto data = ShaderModule::ReadFile(filename);
    if (data.empty())
        throw std::runtime_error("Failed to open file: " + filename);
    return data;
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// =============================================================================
// MAIN APPLICATION CLASS
// =============================================================================
class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // --- Window ---
    SDL_Window* window = nullptr;

    // --- Vulkan Core ---
    vk::Instance               m_instance;
    vk::DebugUtilsMessengerEXT m_debugMessenger;
    vk::PhysicalDevice         m_physicalDevice;
    vk::Device                 m_device;
    vk::SurfaceKHR             m_surface;

    // --- Queues ---
    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;
    vk::Queue m_computeQueue;        // <<< Async Compute Queue

    uint32_t m_graphicsFamily = 0;
    uint32_t m_presentFamily  = 0;
    uint32_t m_computeFamily  = 0;
    bool     m_hasAsyncCompute = false;

    // --- Swapchain ---
    vk::SwapchainKHR             m_swapChain;
    std::vector<vk::Image>       m_swapChainImages;
    vk::Format                   m_swapChainImageFormat;
    vk::Extent2D                 m_swapChainExtent;
    std::vector<vk::ImageView>   m_swapChainImageViews;
    std::vector<vk::Framebuffer> m_swapChainFramebuffers;   // untuk pass KOMPOSIT

    // --- Fase 3 langkah 1: target warna HDR offscreen ---------------------
    // Main pass tidak lagi menggambar langsung ke swapchain. Ia menggambar ke
    // sini (rgba16f), lalu pass komposit yang memampatkannya ke layar.
    //
    // Ini prasyarat struktural, bukan sekadar tonemap: TAA, bloom, dan seluruh
    // post-processing berikutnya perlu MEMBACA warna frame sebelum ia sampai ke
    // layar. Selama main pass menulis langsung ke swapchain, tidak satu pun dari
    // itu mungkin.
    vk::Image          m_hdrImage;
    vk::DeviceMemory   m_hdrMemory;
    vk::ImageView      m_hdrImageView;
    vk::Framebuffer    m_hdrFramebuffer;
    vk::RenderPass     m_compositeRenderPass;
    vk::DescriptorSetLayout m_compositeDSL;
    vk::DescriptorPool      m_compositePool;
    vk::DescriptorSet       m_compositeSet;
    vk::PipelineLayout      m_compositePipelineLayout;
    vk::Pipeline            m_compositePipeline;
    bool                    m_swapchainIsSrgb = false;

    // --- TAA ---------------------------------------------------------------
    // Dua history warna, dipakai bergantian: frame ini membaca yang satu dan
    // menulis yang lain. Composite membaca yang baru saja ditulis, jadi
    // descriptor komposit juga ada dua.
    std::array<vk::Image,2>        m_taaImage{};
    std::array<vk::DeviceMemory,2> m_taaMemory{};
    std::array<vk::ImageView,2>    m_taaView{};
    std::array<vk::DescriptorSet,2> m_compositeSets{};
    vk::DescriptorSetLayout m_taaDSL;
    vk::DescriptorPool      m_taaPool;
    std::array<vk::DescriptorSet,2> m_taaSets{};
    vk::PipelineLayout      m_taaPipelineLayout;
    vk::Pipeline            m_taaPipeline;
    uint32_t                m_taaParity = 0;
    uint32_t                m_taaFrame  = 0;
    float                   m_lastDeltaTime = 1.0f / 60.0f;
    // 1 = frame rate cukup untuk TAA menumpuk, 0 = tidak. Dihitung dari
    // deltaTime, dipakai menskalakan jitter DAN bobot blend sekaligus.
    float                   m_taaHealth = 0.0f;
    glm::vec2               m_jitterPx{0.0f};      // jitter frame ini, satuan piksel
    glm::mat4               m_prevViewProjUnjit{1.0f};
    bool                    m_hasPrevVP = false;

    // --- Depth ---
    vk::Image        m_depthImage;
    vk::DeviceMemory m_depthImageMemory;
    vk::ImageView    m_depthImageView;

    // --- Shadow Map ---
    // Member shadow map lama DIHAPUS — resource-nya kini milik m_shadowAAA.

    // --- Main Render Pass ---
    vk::RenderPass     m_renderPass;
    vk::PipelineLayout m_pipelineLayout;
    vk::Pipeline       m_graphicsPipeline;
    // Langit analitik (v36). Memakai m_pipelineLayout dan m_renderPass yang
    // sama dengan geometri — nol descriptor baru, nol render pass baru.
    vk::Pipeline       m_skyPipeline = nullptr;
    vk::DescriptorSetLayout m_descriptorSetLayout;
    vk::DescriptorPool      m_descriptorPool;
    std::vector<vk::DescriptorSet> m_descriptorSetsLevel;
    std::vector<vk::DescriptorSet> m_descriptorSetsFloor;

    // --- Joystick ---
    vk::PipelineLayout m_joystickPipelineLayout = nullptr;
    vk::Pipeline       m_joystickPipeline       = nullptr;
    vk::Buffer         m_joystickVertexBuffer;
    vk::DeviceMemory   m_joystickVertexBufferMemory;
    uint32_t           m_joystickVertexCount = 0;

    // --- Command ---
    vk::CommandPool                  m_commandPool;
    vk::CommandPool                  m_computeCommandPool; // <<< Async Compute
    std::vector<vk::CommandBuffer>   m_commandBuffers;
    std::vector<vk::CommandBuffer>   m_computeCommandBuffers; // <<< Async Compute

    // --- Sync ---
    std::vector<vk::Semaphore> m_imageAvailableSemaphores;
    std::vector<vk::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::Semaphore> m_computeFinishedSemaphores; // <<< Async Compute
    std::vector<vk::Fence>     m_inFlightFences;
    std::vector<vk::Fence>     m_computeFences;             // <<< Async Compute

    // --- Buffers ---
    vk::Buffer       m_vertexBuffer,      m_floorVertexBuffer;
    vk::DeviceMemory m_vertexBufferMemory, m_floorVertexBufferMemory;

    // --- Bola dinamis ---
    // Vertex buffer-nya HOST VISIBLE dan dipetakan permanen: posisinya ditulis
    // ulang tiap frame, jadi menyalin lewat staging buffer tiap frame justru
    // lebih mahal daripada menulis langsung.
    vk::Buffer             m_ballVertexBuffer{}, m_ballIndexBuffer{};
    vk::DeviceMemory       m_ballVertexMemory{}, m_ballIndexMemory{};
    void*                  m_ballMapped   = nullptr;
    uint32_t               m_ballIndexCount = 0;
    std::vector<Vertex>    m_ballVerts;        // salinan CPU, posisi lokal
    std::vector<glm::vec3> m_ballLocalPos;     // posisi lokal murni
    vk::Buffer       m_indexBuffer,       m_floorIndexBuffer;
    vk::DeviceMemory m_indexBufferMemory,  m_floorIndexBufferMemory;

    std::vector<vk::Buffer>       m_uniformBuffersLevel, m_uniformBuffersFloor;
    std::vector<vk::DeviceMemory> m_uniformBuffersMemoryLevel, m_uniformBuffersMemoryFloor;
    std::vector<void*>            m_uniformBuffersMappedLevel, m_uniformBuffersMappedFloor;

    // --- Frustum Culling Data ---
    std::vector<TileInfo> m_tiles;

    // Jarak yang perlu dicakup cascade, diturunkan dari geometri sungguhan saat
    // level dibangun. 0 = belum dihitung, jalur lama yang dipakai.
    float                 m_shadowFitDist = 0.0f;
    Frustum               m_frustum;
    std::mutex            m_frustumMutex;

    // --- Cached draw state after culling ---
    struct CulledDraw {
        uint32_t firstIndex;
        uint32_t indexCount;
    };
    std::vector<CulledDraw> m_visibleTiles;
    std::mutex              m_visibleMutex;
    std::mutex m_inputMutex; // melindungi m_camera + m_moveInputVector

    // --- Frame tracking ---
    uint32_t m_currentFrame   = 0;
    uint32_t m_maxFrames      = 0;
    uint32_t m_levelIndexCount= 0;

    // --- Camera ---
    glm::vec3 m_cameraPos         = glm::vec3(0.0f, Config::PLAYER_HEIGHT, 0.0f);
    glm::vec3 m_cameraFront       = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 m_cameraUp          = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::quat m_cameraOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    float     m_cameraYaw   = 0.0f;
    float     m_cameraPitch = 0.0f;

    // --- Game state (protected by m_gameStateMutex) ---
    std::mutex  m_gameStateMutex;
    glm::mat4   m_shadowMVP_Level;
    glm::mat4   m_shadowMVP_Floor;
    glm::mat4   m_viewMatrix;
    glm::mat4   m_projMatrix;
    glm::vec3   m_lightPos;

    // --- Joystick ---
    glm::vec2 m_joystickCenter;
    float     m_joystickRadius     = 0.0f;
    glm::vec2 m_joystickKnobPos;
    float     m_joystickKnobRadius = 0.0f;
    glm::vec2 m_moveInputVector;

    std::map<SDL_FingerID, TouchPoint> m_activeTouches;

    // --- Thread Pool + Job System (dari main.cpp) ---
    std::unique_ptr<ThreadPool> m_threadPool;
    std::unique_ptr<JobSystem>  m_jobSystem;

    // --- Camera class (dari main.cpp) ---
    FirstPersonCamera m_camera;

    // --- Vulkan Memory Allocator (dari main.cpp) ---
    std::unique_ptr<VulkanMemoryAllocator> m_vma;

    // --- FrustumCullingManager draw calls (extended dari main.cpp) ---
    std::vector<DrawCall>   m_levelDrawCalls;    // per-face draw calls
    std::vector<DrawCall>   m_mergedDrawCalls;   // setelah merge optimization
    std::vector<uint32_t>   m_visibleDrawIndices;

    // --- Render graph resources untuk automatic barriers (dari main.cpp) ---
    // FIX: per-frame array agar setiap frame melacak currentLayout-nya sendiri.
    // Single shared resource menyebabkan oldLayout mismatch saat image di-swap antar frame.
    RenderResource m_depthRenderResource;
    std::vector<RenderResource*> m_allRenderResources;

    // --- Shadow AAA (CSM + PCSS + Temporal + Bilateral) ---
    ShadowAAA::Pipeline   m_shadowAAA;
    ShadowAAA::GpuProfiler m_gpuProfiler;
    // Jumlah draw call yang benar-benar dikirim per cascade setelah culling.
    std::array<uint32_t, ShadowAAA::Cfg::NUM_CASCADES> m_shadowDrawCalls{};
    // v49: jumlah INDEKS yang benar-benar dikirim ke tiap cascade. Draw call
    // saja menyesatkan — satu draw call bisa berisi 10 indeks atau 100.000.
    // Ini angka yang menentukan berapa segitiga yang benar-benar diraster.
    std::array<uint32_t, ShadowAAA::Cfg::NUM_CASCADES> m_shadowIndices{};

    // --- Timeline semaphore untuk compute sync (dari main.cpp) ---
    TimelineSemaphore m_computeTimeline;

    // --- Per-frame thread data untuk parallel recording (dari main.cpp) ---
    std::vector<PerFrameThreadData> m_perFrameThreadData;

    // --- Indirect draw buffer (GPU-driven, dari main.cpp) ---
    IndirectDrawBuffer m_indirectDrawBuffer;

    // --- Halton sequence untuk TAA jitter (dari main.cpp) ---
    uint32_t m_temporalFrameIndex = 0;
    glm::vec2 m_jitterOffset      = glm::vec2(0.0f);
    
    // --- Game loop timing (ganti static variable) ---
    std::chrono::high_resolution_clock::time_point m_startTime;
    float m_lastGameTime    = 0.0f;
    // -1e9 supaya `time - m_lastLightUpdate > 2.0f` PASTI lolos di frame
    // PERTAMA. Nilai awal 0.0f dulu membuat ambang itu baru terlampaui pada
    // detik ke-2 — di 0,5 FPS itu satu frame penuh yang dirender dengan sudut
    // matahari yang belum pernah dihitung.
    float m_lastLightUpdate = -1.0e9f;

    // Nilai awal harus sudah BENAR, bukan nol. Nol berarti azimut 0 derajat
    // (lightPos = 20, 12, 0), sedangkan SUN_FREEZE_ANGLE_DEG = 45 memberi
    // (14.14, 12, 14.14) — selisih 45 derajat. Itulah "bayangan lurus dulu,
    // baru miring" yang terlihat saat aplikasi baru jalan: frame pertama
    // memakai azimut 0, lalu melompat ke 45 begitu update pertama berjalan.
    //
    // Dengan m_lastLightUpdate = -1e9 lompatan itu sebenarnya sudah tidak
    // mungkin terjadi, tapi nilai awal ini tetap dipasang supaya tidak ada satu
    // pun jalur yang bisa membaca sudut nol — termasuk kalau nanti ada kode
    // lain yang menyentuh lightPos sebelum updateGameState() pertama.
    float m_cachedLightAngle = ShadowAAA::Cfg::SUN_FREEZE
                             ? glm::radians(ShadowAAA::Cfg::SUN_FREEZE_ANGLE_DEG)
                             : 0.0f;

    // --- Async frame futures ---
    std::future<void> m_gameStateFuture;
    std::future<void> m_cullingFuture;
    std::future<void> m_uboFuture;
    JobHandle         m_cullingJobHandle;
    JobHandle         m_uboJobHandle;

    // ==========================================================================
    // INIT
    // ==========================================================================
    void initWindow() {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw std::runtime_error("Gagal SDL_Init");
        // v73: di Android ukuran window DITENTUKAN SISTEM, bukan oleh kita.
        // Meminta 800x600 di sana tidak salah tapi diabaikan, dan yang berbahaya
        // adalah kalau kode di bawahnya terus memakai angka 800x600 untuk aspek
        // rasio sementara layar sebenarnya 1080x2340 — gambarnya akan gepeng.
        //
        // Jadi window dibuat, lalu ukuran SEBENARNYA dibaca kembali.
        window = SDL_CreateWindow(
            Config::APP_NAME.c_str(),
            static_cast<int>(Config::WINDOW_WIDTH),
            static_cast<int>(Config::WINDOW_HEIGHT),
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!window) throw std::runtime_error(
            std::string("Gagal SDL_CreateWindow: ") + SDL_GetError());

        int gotW = 0, gotH = 0;
        SDL_GetWindowSizeInPixels(window, &gotW, &gotH);
        if (gotW > 0 && gotH > 0) {
            g_windowWidth  = static_cast<uint32_t>(gotW);
            g_windowHeight = static_cast<uint32_t>(gotH);
        }
        SDL_SetWindowRelativeMouseMode(window, false);
        LOG_INFO("App", "Window created " + std::to_string(g_windowWidth)
                 + "x" + std::to_string(g_windowHeight));
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();

        m_maxFrames = static_cast<uint32_t>(m_swapChainImages.size());

        createDepthResources();
        // createShadowResources / RenderPass / Pipeline / Framebuffer DIHAPUS:
        // digantikan sepenuhnya oleh ShadowAAA::Pipeline (cascade array 4 layer).
        createImageViews();
        createHdrTarget();            // target warna offscreen (Fase 3)
        createRenderPass();           // main pass -> HDR
        createCompositeRenderPass();  // HDR -> swapchain
        createGraphicsPipeline();
        createSkyPipeline();          // langit + berkas cahaya (v36)
        createJoystickPipeline();
        // createFramebuffers() dan createCompositePipeline() dipanggil SETELAH
        // ShadowAAA::init, karena pipeline komposit meminjam sampler miliknya.
        createCommandPool();
        createComputeCommandPool();  // <<< Async Compute

        // === SHADOW AAA: init sebelum createUniformBuffers() ===
        // createUniformBuffers() butuh image view final dari modul ini untuk
        // ditulis ke descriptor binding 1.
        {
            ShadowAAA::InitInfo si{};
            si.physicalDevice = m_physicalDevice;
            si.device         = m_device;
            si.graphicsQueue  = m_graphicsQueue;
            si.commandPool    = m_commandPool;
            si.screenExtent   = m_swapChainExtent;
            si.framesInFlight = m_maxFrames;
            si.shaderDir      = "";   // ubah ke "shaders/" kalau .spv ada di subfolder

            si.vertexBinding  = Vertex::getBindingDescription();
            auto attrs        = Vertex::getAttributeDescriptions();
            si.vertexAttributes.assign(attrs.begin(), attrs.end());

            m_shadowAAA.init(si);
            // Jumlah cascade dibaca dari konstantanya, bukan ditulis tangan.
            // Sebelumnya baris ini mencetak "4-cascade" padahal NUM_CASCADES
            // sudah 6 sejak v9 — laporan yang berbohong tentang kodenya sendiri.
            LOG_INFO("ShadowAAA", "CSM " +
                std::to_string(ShadowAAA::Cfg::NUM_CASCADES) + "-cascade " +
                std::to_string(ShadowAAA::Cfg::CASCADE_RES) + "^2 + PCSS + temporal aktif");
            // Diagnostik penting: kalau hardware PCF TIDAK tersedia, posisi tepi
            // bayangan akan terkunci ke texel dan terlihat "sobek". Shader lalu
            // memakai bilinear manual (4x fetch per tap) untuk memperbaikinya.
            LOG_INFO("ShadowAAA", m_shadowAAA.hardwarePcfAvailable()
                ? "Hardware 2x2 PCF: TERSEDIA (jalur murah)"
                : "Hardware 2x2 PCF: TIDAK ADA -> pakai bilinear manual di shader");

            // Diagnostik performa. Laju piksel terukur (31 Mpiksel/detik di pass
            // cascade) ada di rentang rasterizer PERANGKAT LUNAK, bukan GPU.
            // Baris ini memastikannya: kalau namanya mengandung llvmpipe,
            // lavapipe, SwiftShader, atau tipenya CPU, berarti tidak ada
            // akselerasi GPU sama sekali dan itulah batas performanya.
            {
                auto props = m_physicalDevice.getProperties();
                const char* t = "lainnya";
                switch (props.deviceType) {
                    case vk::PhysicalDeviceType::eDiscreteGpu:   t = "GPU diskrit"; break;
                    case vk::PhysicalDeviceType::eIntegratedGpu: t = "GPU terintegrasi"; break;
                    case vk::PhysicalDeviceType::eVirtualGpu:    t = "GPU virtual"; break;
                    case vk::PhysicalDeviceType::eCpu:           t = "CPU (rasterizer perangkat lunak!)"; break;
                    default: break;
                }
                LOG_INFO("GPU", std::string("Device: ") + props.deviceName.data()
                                + " | tipe: " + t);
            }

            m_gpuProfiler.init(m_physicalDevice, m_device, m_maxFrames);
            LOG_INFO("GPU", m_gpuProfiler.enabled()
                ? "Timestamp query aktif — laporan waktu per pass tiap 15 frame"
                : "Timestamp query TIDAK didukung device ini");
        }

        // Setelah ShadowAAA siap: framebuffer HDR dan pipeline komposit.
        // Pipeline komposit meminjam sampler milik ShadowAAA, jadi urutannya
        // tidak boleh dibalik.
        createFramebuffers();
        createTaaResources();        // sebelum komposit: komposit membaca history TAA
        createCompositePipeline();

        // v41: peta HARUS diperbesar SEBELUM geometri apa pun dibangun.
        // createLevelBuffers() dan createFloorBuffers() sama-sama membaca
        // levelMap, dan m_shadowFitDist diturunkan dari AABB hasilnya — kalau
        // urutannya terbalik, yang terbangun adalah peta lama dan pembesarannya
        // tidak berpengaruh apa-apa.
        checkStaleShaderBinaries();

        enlargeLevelMap();

        createLevelBuffers();
        createFloorBuffers();
        createBallBuffers();
        createJoystickBuffers();
        createUniformBuffers();
        createCommandBuffers();
        createComputeCommandBuffers(); // <<< Async Compute
        createSyncObjects();
        updateJoystickLayout();

        m_threadPool = std::make_unique<ThreadPool>();
        m_jobSystem  = std::make_unique<JobSystem>(*m_threadPool);
        m_vma        = std::make_unique<VulkanMemoryAllocator>(m_physicalDevice, m_device);
        
        m_startTime = std::chrono::high_resolution_clock::now();

        // Inisialisasi Halton sequence untuk TAA jitter
        HaltonSequence::initialize();

        // Setup camera aspect ratio
        m_camera.SetAspectRatio(
            m_swapChainExtent.width / (float)m_swapChainExtent.height);

        // Setup render resources untuk RenderGraph automatic barriers.
        // m_shadowRenderResources DIHAPUS: layout cascade array kini diurus
        // sepenuhnya oleh render pass di dalam ShadowAAA::Pipeline.
        m_depthRenderResource.aspect        = vk::ImageAspectFlagBits::eDepth;
        m_depthRenderResource.name          = "DepthBuffer";
        m_depthRenderResource.image         = m_depthImage;
        m_depthRenderResource.imageView     = m_depthImageView;
        m_allRenderResources = { &m_depthRenderResource };

        LOG_INFO("App", "Vulkan initialized. Async compute: " +
            std::string(m_hasAsyncCompute ? "YES" : "NO (using graphics queue)") +
            " | JobSystem: " + std::to_string(m_threadPool->threadCount()) + " workers" +
            " | VMA: active | RenderGraph: active");
    }

    // ==========================================================================
    // ASYNC COMPUTE: Buat command pool & buffers untuk compute queue
    // ==========================================================================
    void createComputeCommandPool() {
        uint32_t qf = m_hasAsyncCompute ? m_computeFamily : m_graphicsFamily;
        vk::CommandPoolCreateInfo info{};
        info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        info.queueFamilyIndex = qf;
        m_computeCommandPool = m_device.createCommandPool(info);
        LOG_INFO("Compute", "Compute command pool created (family=" + std::to_string(qf) + ")");
    }

    void createComputeCommandBuffers() {
        m_computeCommandBuffers.resize(m_maxFrames);
        vk::CommandBufferAllocateInfo info{};
        info.commandPool        = m_computeCommandPool;
        info.level              = vk::CommandBufferLevel::ePrimary;
        info.commandBufferCount = (uint32_t)m_computeCommandBuffers.size();
        m_computeCommandBuffers = m_device.allocateCommandBuffers(info);
    }

    // Record sebuah compute command (demonstrasi: memory barrier + timestamp)
    // Di sini Anda bisa menambahkan dispatch compute shader untuk GPU-side culling.
    void recordComputeCommands(uint32_t frameIndex) {
        auto& cb = m_computeCommandBuffers[frameIndex];
        cb.reset();
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        cb.begin(beginInfo);

        // ---- Slot untuk GPU-side compute shader (misal: indirect draw build, particle) ----
        // Contoh: pipeline barrier untuk sinkronisasi setelah compute selesai
        vk::MemoryBarrier barrier{};
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexShader,
            vk::DependencyFlags{},
            1, &barrier, 0, nullptr, 0, nullptr);

        cb.end();
    }

    // Submit compute work SEBELUM graphics, gunakan semaphore untuk sinkronisasi
    void submitComputeWork(uint32_t frameIndex) {
        recordComputeCommands(frameIndex);

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &m_computeCommandBuffers[frameIndex];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &m_computeFinishedSemaphores[frameIndex];

        vk::Queue computeQ = m_hasAsyncCompute ? m_computeQueue : m_graphicsQueue;
        computeQ.submit(submitInfo, m_computeFences[frameIndex]);
    }

    // ==========================================================================
    // FRUSTUM CULLING — dijalankan di thread pool (TaskPriority::High)
    // ==========================================================================
    void updateFrustumCulling(const glm::mat4& viewProj) {
        std::lock_guard<std::mutex> lk(m_frustumMutex);
        m_frustum.ExtractFromMatrix(viewProj); // ExtractFromMatrix dari enhanced Frustum

        // === PASS 1: Tile-level culling (kasar, cepat) ===
        std::vector<CulledDraw> visible;
        visible.reserve(m_tiles.size());
        for (auto& tile : m_tiles) {
            tile.visible = m_frustum.TestAABB(tile.aabb);
            if (tile.visible)
                visible.push_back({ tile.firstIndex, tile.indexCount });
        }
        {
            std::lock_guard<std::mutex> vlk(m_visibleMutex);
            m_visibleTiles = std::move(visible);
        }

        // === PASS 2: Per-face culling dengan FrustumCullingManager (halus) ===
        // Jalankan jika levelDrawCalls sudah diisi
        if (!m_levelDrawCalls.empty()) {
            std::vector<uint32_t> visibleFaceIndices;
            uint32_t visCount = FrustumCullingManager::CullDrawCalls(
                m_levelDrawCalls, m_frustum, visibleFaceIndices);

            // Merge draw calls berurutan untuk mengurangi API call overhead
            std::vector<DrawCall> merged;
            FrustumCullingManager::MergeConsecutiveDrawCalls(
                m_levelDrawCalls, visibleFaceIndices, merged);

            {
                std::lock_guard<std::mutex> vlk(m_visibleMutex);
                m_mergedDrawCalls      = std::move(merged);
                m_visibleDrawIndices   = std::move(visibleFaceIndices);
            }

            LOG_DEBUG("Culling",
                "Faces: " + std::to_string(visCount) + "/" +
                std::to_string(m_levelDrawCalls.size()) +
                " | Merged DC: " + std::to_string(m_mergedDrawCalls.size()) +
                " | Tiles: " + std::to_string(m_visibleTiles.size()) + "/" +
                std::to_string(m_tiles.size()));
        } else {
            LOG_DEBUG("Culling", "Visible tiles: " +
                std::to_string(m_visibleTiles.size()) + " / " +
                std::to_string(m_tiles.size()));
        }
    }

    // ==========================================================================
    // GAME STATE UPDATE — thread-safe, dijalankan di worker thread
    // Snapshot input di awal lalu lepas lock, sehingga main thread tidak
    // terkunci lama saat handleInput memproses event berikutnya.
    // ==========================================================================
    void updateGameState(uint32_t currentImage) {
    
        // =========================================================================
        // SNAPSHOT INPUT STATE — lock singkat, lepas segera
        // Main thread menulis m_camera + m_moveInputVector via m_inputMutex.
        // Worker thread membaca snapshot-nya di sini agar tidak terjadi data race.
        // =========================================================================
        glm::vec2 moveInput;
        glm::vec3 camPos;
        glm::quat camOrientation;
        float     camYaw, camPitch;
        {
            std::lock_guard<std::mutex> lk(m_inputMutex);
            moveInput      = m_moveInputVector;
            camPos         = m_camera.GetPosition();
            camOrientation = m_camera.GetOrientation();
            camYaw         = m_camera.GetYaw();
            camPitch       = m_camera.GetPitch();
        }
    
        // =========================================================================
        // TIMING
        // =========================================================================
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(
            currentTime - m_startTime).count();
    
        float rawDelta = time - m_lastGameTime;
        if (rawDelta <= 0.0f) rawDelta = 0.001f;

        // deltaTime dipotong 0,1 s untuk GERAKAN — itu benar dan harus tetap:
        // tanpa itu satu frame lambat membuat pemain melompat jauh.
        //
        // Tapi potongan itu TIDAK BOLEH ikut ke logika temporal. Peredaman
        // riwayat TAA dan bayangan memakai exp(-dt/tau); memberinya 0,1 detik
        // saat frame nyatanya 2 detik membuatnya meleset dua puluh kali, dan
        // riwayat bertahan belasan detik. Itu ghosting yang tersisa setelah dua
        // perbaikan sebelumnya — keduanya benar rumusnya, tapi diberi angka
        // yang sudah dipalsukan di hulu.
        //
        // Jadi dua nilai dipisah: deltaTime untuk gerakan, rawDelta untuk waktu.
        float deltaTime = (rawDelta > 0.1f) ? 0.1f : rawDelta;
        m_lastGameTime = time;
    
        // =========================================================================
        // MOVEMENT — menggunakan snapshot, tidak perlu lock
        // =========================================================================
        glm::vec3 moveDir(0.0f);
        if (glm::length(moveInput) > Config::JOYSTICK_DEADZONE) {
            // Hitung forward/right dari snapshot orientasi kamera
            glm::vec3 camForward = camOrientation * glm::vec3(0.0f, 0.0f, -1.0f);
            glm::vec3 camRight   = camOrientation * glm::vec3(1.0f, 0.0f,  0.0f);
    
            // Proyeksikan ke bidang horizontal
            camForward.y = 0.0f;
            camRight.y   = 0.0f;
    
            if (glm::length(camForward) > 0.001f) camForward = glm::normalize(camForward);
            if (glm::length(camRight)   > 0.001f) camRight   = glm::normalize(camRight);
    
            moveDir = camRight   *  moveInput.x
                    + camForward * -moveInput.y;
        }
    
        if (glm::length(moveDir) > 0.0f) {
            moveDir = glm::normalize(moveDir);
            glm::vec3 newPos = camPos + moveDir * (Config::PLAYER_SPEED * deltaTime);
    
            if (!checkCollision(newPos)) {
                camPos = newPos;
            } else {
                // Slide di sumbu X
                glm::vec3 slideX = camPos;
                slideX.x = newPos.x;
                if (!checkCollision(slideX)) camPos.x = newPos.x;
    
                // Slide di sumbu Z
                glm::vec3 slideZ = camPos;
                slideZ.z = newPos.z;
                if (!checkCollision(slideZ)) camPos.z = newPos.z;
            }
        }
        camPos.y = Config::PLAYER_HEIGHT;
    
        // =========================================================================
        // TULIS BALIK POSISI KE KAMERA — lock singkat
        // Hanya SetPosition yang perlu di-write-back; orientasi tidak berubah
        // di sini (diubah hanya oleh handleInput via Rotate).
        // =========================================================================
        {
            std::lock_guard<std::mutex> lk(m_inputMutex);
            m_camera.SetPosition(camPos);
            // Sync backward-compat fields
            m_cameraPos         = camPos;
            m_cameraOrientation = camOrientation;
            m_cameraYaw         = camYaw;
            m_cameraPitch       = camPitch;
        }
    
        // =========================================================================
        // MATRICES
        // =========================================================================
        glm::mat4 view  = glm::mat4_cast(glm::conjugate(camOrientation))
                        * glm::translate(glm::mat4(1.0f), -camPos);
    
        glm::mat4 proj  = glm::perspective(
            glm::radians(Config::CAMERA_FOV),
            m_swapChainExtent.width / (float)m_swapChainExtent.height,
            Config::CAMERA_NEAR_PLANE,
            Config::CAMERA_FAR_PLANE);
        proj[1][1] *= -1.0f; // Vulkan convention
    
        glm::mat4 model = glm::mat4(1.0f);
    
        // =========================================================================
        // LIGHT — semi-statis, update setiap 2 detik
        // m_lastLightUpdate dan m_cachedLightAngle hanya diakses dari job Critical
        // (satu job sekaligus), jadi tidak perlu mutex.
        // =========================================================================
        if (time - m_lastLightUpdate > 2.0f) {
            // SUN_FREEZE: sudut matahari dipaku, tidak lagi turunan dari waktu.
            //
            // Ini alat ukur, bukan fitur tampilan. Sudut matahari selama ini
            // dihitung dari `time` = detik SEJAK APLIKASI START. Akibatnya dua
            // screenshot dari dua run berbeda hampir tidak pernah punya matahari
            // di posisi yang sama, dan membandingkan bayangan di antara keduanya
            // tidak sah — bayangan yang bergeser bisa berarti perubahan kode,
            // bisa juga cuma matahari yang sudah pindah.
            //
            // Terukur pada laju orbit yang berlaku (periode 6 jam = 16,67 mderajat
            // per detik, caster 1,16 m, elevasi 31 derajat): selisih 40 detik
            // saja sudah menggeser ujung bayangan 2,25 cm = 10,6 piksel. Selisih
            // 17,7 menit menggesernya 281 piksel — lebih lebar dari seluruh
            // bayangan yang sedang diukur.
            //
            // Dengan SUN_FREEZE = true, screenshot dari run mana pun bisa
            // ditumpuk langsung. Set false untuk mengembalikan matahari berputar.
            m_cachedLightAngle = ShadowAAA::Cfg::SUN_FREEZE
                ? glm::radians(ShadowAAA::Cfg::SUN_FREEZE_ANGLE_DEG)
                : time * (6.28318530718f / ShadowAAA::Cfg::SUN_ORBIT_PERIOD_SEC);
            m_lastLightUpdate  = time;
        }
    
        updateBall(time);

        glm::vec3 lp;
        lp.y = 12.0f;
        lp.x = 20.0f * cosf(m_cachedLightAngle);
        lp.z = 20.0f * sinf(m_cachedLightAngle);
    
        // ---- Jitter TAA --------------------------------------------------
        // Proyeksi digeser kurang dari satu piksel tiap frame mengikuti deret
        // Halton(2,3). Deret ini dipilih karena sebarannya rata pada setiap
        // panjang awalan — berhenti di frame ke berapa pun, sampelnya tetap
        // tersebar merata. Deret acak biasa bisa menggerombol.
        //
        // Jitter DIBAKAR ke dalam matriks, bukan dikirim terpisah ke shader:
        // ia cuma translasi di clip space, jadi mengalikannya di CPU membuat
        // seluruh shader tidak perlu tahu-menahu.
        {
            auto halton = [](uint32_t i, uint32_t b) {
                float f = 1.0f, r = 0.0f;
                while (i > 0) { f /= b; r += f * (i % b); i /= b; }
                return r;
            };
            // KESEHATAN TAA — jitter diskalakan menurut frame rate.
            //
            // Ini bukan kehati-hatian berlebihan, ini koreksi kesalahan nyata.
            // TAA menukar ketajaman satu frame dengan penumpukan banyak frame:
            // tiap frame sengaja digeser sampai setengah piksel, dan penumpukan
            // itu yang mengembalikan ketajamannya. Kalau penumpukannya tidak
            // pernah terjadi, yang tersisa cuma pergeserannya — jadi TAA di FPS
            // rendah BUKAN netral, ia merugikan. Terukur: siluet tetap 1 piksel
            // transisi (tidak ada anti-aliasing sama sekali) tapi tepinya kini
            // bergoyang tiap frame.
            //
            // Ambangnya dari deltaTime langsung: penuh di atas 24 FPS, nol di
            // bawah 8 FPS, melandai di antaranya. Di PC nanti selalu penuh dan
            // baris ini tidak berpengaruh; di llvmpipe ia mematikan jitter
            // sendiri tanpa perlu mengubah konstanta apa pun.
            {
                // rawDelta, BUKAN deltaTime.
                //
                // deltaTime dipotong 0,1 s untuk gerakan, jadi ambang ini
                // mengira frame paling lambat pun cuma 10 FPS. Di 0,5 FPS
                // health jadi 0,216 — padahal ambangnya dirancang mematikan
                // TAA total di bawah 8 FPS.
                //
                // Akibatnya yang terburuk dari dua dunia: jitter tetap
                // menggeser piksel dan history tetap dicampur, tapi terlalu
                // lemah untuk meng-anti-alias apa pun, dan cukup kuat untuk
                // membuat gambar goyang.
                const float dtLo = 1.0f / 24.0f, dtHi = 1.0f / 8.0f;
                float t = glm::clamp((rawDelta - dtLo) / (dtHi - dtLo), 0.0f, 1.0f);
                m_taaHealth = 1.0f - (t * t * (3.0f - 2.0f * t));
            }

            if (Config::TAA_ENABLED) {
                uint32_t k = (m_taaFrame % Config::TAA_JITTER_COUNT) + 1;
                m_jitterPx = glm::vec2(halton(k, 2) - 0.5f, halton(k, 3) - 0.5f)
                           * m_taaHealth;
            } else {
                m_jitterPx = glm::vec2(0.0f);
                m_taaHealth = 0.0f;
            }
            ++m_taaFrame;
        }

        const float jx =  2.0f * m_jitterPx.x / (float)m_swapChainExtent.width;
        const float jy =  2.0f * m_jitterPx.y / (float)m_swapChainExtent.height;

        glm::mat4 jitterM(1.0f);
        jitterM[3][0] = jx;
        jitterM[3][1] = jy;

        const glm::mat4 viewProjUnjit = proj * view;
        const glm::mat4 viewProjJit   = jitterM * viewProjUnjit;

        // prevViewProj dibangun ulang dengan jitter frame INI, bukan jitter
        // frame lalu. Dengan begitu suku jitter saling menghapus saat pre-pass
        // menghitung velocity, sehingga motion vector-nya bebas jitter — dan
        // TAA tidak mengejar goyangan yang ia buat sendiri.
        const glm::mat4 prevViewProjJit =
            jitterM * (m_hasPrevVP ? m_prevViewProjUnjit : viewProjUnjit);
        m_prevViewProjUnjit = viewProjUnjit;
        m_hasPrevVP = true;

        // Matriks light lama dipertahankan hanya untuk field UBO lightMVP
        // (kompatibilitas layout). Bayangan sesungguhnya kini dihitung ShadowAAA.
        glm::mat4 lightView = glm::lookAt(lp, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 lightProj = glm::ortho(-25.0f, 25.0f, -25.0f, 25.0f, 1.0f, 40.0f);
        lightProj[1][1] *= -1.0f;
        glm::mat4 lightVP = lightProj * lightView;

        // =========================================================================
        // SHADOW AAA — hitung 4 cascade + isi UBO compute untuk slot frame ini.
        // Dipanggil dari job Critical, satu kali per frame, jadi state sekuensial
        // di dalam modul (prevViewProj, parity ping-pong) aman.
        // =========================================================================
        {
            ShadowAAA::FrameInput fi{};
            fi.cameraPos     = camPos;
            fi.cameraForward = camOrientation * glm::vec3(0.0f, 0.0f, -1.0f);
            fi.cameraRight   = camOrientation * glm::vec3(1.0f, 0.0f,  0.0f);
            fi.cameraUp      = camOrientation * glm::vec3(0.0f, 1.0f,  0.0f);
            fi.fovYRadians   = glm::radians(Config::CAMERA_FOV);
            fi.aspect        = m_swapChainExtent.width / (float)m_swapChainExtent.height;
            fi.nearZ         = Config::CAMERA_NEAR_PLANE;
            fi.farZ          = Config::CAMERA_FAR_PLANE;
            fi.shadowFitDist = m_shadowFitDist;

            // v40: selubung SELURUH lintasan bola, bukan posisi sesaatnya.
            // Bola bergerak di sumbu Z sejauh +-BALL_TRAVEL dari BALL_Z, jadi
            // kotak ini konstan sepanjang permainan — cascade yang menyentuhnya
            // akan selalu dirender ulang, sisanya boleh di-cache tanpa harus
            // mengevaluasi ulang apa pun tiap frame.
            if (Config::BALL_ENABLED) {
                const float R = Config::BALL_RADIUS;
                fi.hasDynamic = true;
                fi.dynamicMin = glm::vec3(Config::BALL_X - R,
                                          Config::BALL_Y - R,
                                          Config::BALL_Z - Config::BALL_TRAVEL - R);
                fi.dynamicMax = glm::vec3(Config::BALL_X + R,
                                          Config::BALL_Y + R,
                                          Config::BALL_Z + Config::BALL_TRAVEL + R);
                fi.dynamicRadius = R;
            }
            fi.lightDir      = glm::normalize(lp);   // arah MENUJU cahaya
            fi.view          = view;
            // proj BER-JITTER: pre-pass harus merasterisasi di posisi sub-piksel
            // yang sama dengan main pass, dan csm_resolve merekonstruksi posisi
            // dunia dari depth pre-pass sehingga invViewProj-nya pun harus
            // diturunkan dari matriks yang sama.
            fi.proj          = jitterM * proj;
            fi.prevViewProj  = prevViewProjJit;
            fi.deltaTime     = rawDelta;    // WAKTU NYATA, bukan yang dipotong untuk gerakan
            m_lastDeltaTime  = rawDelta;    // WAKTU NYATA, dipakai push constant TAA
            m_shadowAAA.updateCascades(fi, currentImage);
        }
    
        // =========================================================================
        // HALTON JITTER (TAA prep)
        // m_temporalFrameIndex hanya diakses dari job Critical, tidak perlu mutex.
        // =========================================================================
        uint32_t haltonIdx = m_temporalFrameIndex % HaltonSequence::SAMPLE_COUNT;
        m_jitterOffset     = HaltonSequence::values[haltonIdx];
        m_temporalFrameIndex++;
    
        // =========================================================================
        // TULIS KE SHARED GAME STATE — dilindungi m_gameStateMutex
        // Dibaca oleh recordCommandBuffer di main thread.
        // =========================================================================
        {
            std::lock_guard<std::mutex> lk(m_gameStateMutex);
            m_viewMatrix      = view;
            m_projMatrix      = proj;
            m_lightPos        = lp;
            m_shadowMVP_Level = lightVP * model;
            m_shadowMVP_Floor = lightVP * model;
        }
    
        // =========================================================================
        // FRUSTUM CULLING — dijalankan langsung di sini
        // Aman karena updateGameState sudah di dalam job Critical (satu sekaligus).
        // Tidak ada job lain yang menulis m_frustum / m_mergedDrawCalls bersamaan.
        // =========================================================================
        glm::mat4 viewProj = proj * view;
        updateFrustumCulling(viewProj);
    
        // =========================================================================
        // UBO UPDATE — di-spawn sebagai sub-job High priority
        // Capture by value agar aman walau updateGameState sudah selesai duluan.
        // =========================================================================
        m_uboJobHandle = m_jobSystem->submit(
            [this, view, proj, model, lp, lightVP, camPos, currentImage,
             viewProjJit]() {
    
                UniformBufferObject uboLevel{};
                // Matriks BER-JITTER, sama persis dengan yang dipakai pre-pass.
                // Keduanya harus identik: kalau main pass dan pre-pass merasterisasi
                // di posisi sub-piksel yang berbeda, depth dan normal tidak lagi
                // menggambarkan piksel yang sama dan seluruh rantai bayangan salah.
                uboLevel.mvp              = viewProjJit * model;
                uboLevel.lightMVP         = lightVP * model;
                uboLevel.lightPos         = lp;
                uboLevel.cameraPos        = camPos;
                uboLevel.ambientStrength  = 0.2f;
                uboLevel.diffuseStrength  = 0.7f;
                uboLevel.specularStrength = 0.3f;
                uboLevel.shininess        = 32.0f;
                uboLevel.shadowBias       = 0.005f;
                memcpy(m_uniformBuffersMappedLevel[currentImage],
                       &uboLevel, sizeof(uboLevel));
    
                UniformBufferObject uboFloor = uboLevel;
                memcpy(m_uniformBuffersMappedFloor[currentImage],
                       &uboFloor, sizeof(uboFloor));
    
            }, TaskPriority::High);
    }
    
    // ==========================================================================
    // MAIN LOOP — precompute frame 0, lalu kick N+1 setiap akhir frame
    // ==========================================================================
    void mainLoop() {
        bool running = true;
    
        {
            auto h = m_jobSystem->submit([this]() {
                updateGameState(0);
            }, TaskPriority::Critical);
            m_jobSystem->wait(h);

            // Laporan cascade sekali jalan. Yang paling penting angka "rasio":
            // itu texel shadow map dibagi ukuran satu piksel layar di jarak
            // tersebut. Selama ia sekitar 1, ketajaman yang terlihat sama di
            // semua cascade — seberapa pun jauh jangkauannya.
            LOG_INFO("ShadowAAA", m_shadowAAA.cascadeReport());
        }
    
        while (running) {
            handleInput(running);
            if (!running) break;
            drawFrame();
        }
    
        // TAMBAH: tunggu semua job selesai sebelum device idle
        if (m_cullingJobHandle.isValid()) m_jobSystem->wait(m_cullingJobHandle);
        if (m_uboJobHandle.isValid())     m_jobSystem->wait(m_uboJobHandle);
        m_jobSystem->waitAll();
        m_threadPool->waitAll();
    
        m_device.waitIdle();
    }
    
    // ==========================================================================
    // DRAW FRAME — 4-Submit Pattern (dari main.cpp FC6 style)
    //
    // Submit 1: Async Compute (compute queue / graphics queue fallback)
    //           Sinyal: m_computeTimeline (timeline) + m_computeFinishedSemaphores (binary)
    // Submit 2: Shadow Pass (graphics queue)
    //           Wait: m_computeFinishedSemaphores, Sinyal: shadowPassDone (inline)
    // Submit 3: Main Render Pass (graphics queue)
    //           Wait: imageAvailable + shadowPassDone, Sinyal: renderFinished
    // Submit 4: Precompute frame N+1 (thread pool, async — tidak block present)
    //
    // Setiap submit menggunakan TimelineSubmitInfo untuk mixed binary+timeline semaphore
    // ==========================================================================
    void drawFrame() {
        const uint32_t f = m_currentFrame;
    
        (void)m_device.waitForFences(1, &m_computeFences[f], VK_TRUE, UINT64_MAX);
        (void)m_device.resetFences(1, &m_computeFences[f]);
        submitComputeWork(f);
    
        (void)m_device.waitForFences(1, &m_inFlightFences[f], VK_TRUE, UINT64_MAX);

        // Fence slot ini baru saja ditunggu, jadi timestamp dari pemakaian
        // sebelumnya dijamin sudah siap — pembacaan ini tidak menghentikan GPU.
        m_gpuProfiler.collect(f);
        if (m_gpuProfiler.hasReport()) {
            LOG_INFO("GPU", m_gpuProfiler.takeReport());
            // Efektivitas culling per-cascade: berapa draw call yang benar-benar
            // dikirim ke tiap cascade. Cascade dekat harus jauh lebih kecil
            // daripada cascade jauh — kalau semuanya sama, cullingnya tidak
            // bekerja dan ada yang salah dengan AABB tile atau matriks cascade.
            // v49: BUKTI ANGKA, bukan klaim.
            //
            // Draw call saja menyesatkan — satu draw call bisa berisi 10 indeks
            // atau 100.000. Yang menentukan berapa segitiga yang benar-benar
            // diraster adalah jumlah INDEKS. Baris ini mencetak keduanya berikut
            // persentasenya terhadap level penuh, jadi efektivitas culling
            // terbaca langsung tanpa perlu membandingkan dua build.
            //
            // Kalau semua cascade menunjukkan 100%, cullingnya memang tidak
            // bekerja. Kalau cascade dekat jauh di bawah 100% dan cascade jauh
            // mendekati 100%, ia bekerja persis seperti seharusnya: yang
            // dipangkas justru cascade dekat, tempat resolusi paling padat.
            std::string dc = "culling cascade (indeks dikirim / " +
                             std::to_string(m_levelIndexCount) + "):";
            uint32_t tot = 0;
            for (size_t i = 0; i < m_shadowIndices.size(); ++i) {
                const uint32_t n = m_shadowIndices[i];
                tot += n;
                const int pct = m_levelIndexCount
                              ? static_cast<int>(100.0 * n / m_levelIndexCount) : 0;
                dc += " C" + std::to_string(i) + "=" + std::to_string(n)
                    + "(" + std::to_string(pct) + "%," + std::to_string(m_shadowDrawCalls[i]) + "dc)";
            }
            const uint32_t full = m_levelIndexCount * ShadowAAA::Cfg::NUM_CASCADES;
            dc += " | TOTAL " + std::to_string(tot) + "/" + std::to_string(full)
                + " = " + std::to_string(full ? static_cast<int>(100.0 * tot / full) : 0) + "%";
            LOG_INFO("ShadowAAA", dc);
        }

        (void)m_device.resetFences(1, &m_inFlightFences[f]);
    
        uint32_t imageIndex;
        try {
            vk::ResultValue result = m_device.acquireNextImageKHR(
                m_swapChain, UINT64_MAX,
                m_imageAvailableSemaphores[f], VK_NULL_HANDLE);
            imageIndex = result.value;
        } catch (vk::OutOfDateKHRError&) { return; }
    
        // Tunggu job dari frame sebelumnya
        if (m_cullingJobHandle.isValid()) m_jobSystem->wait(m_cullingJobHandle);
        if (m_uboJobHandle.isValid())     m_jobSystem->wait(m_uboJobHandle);
        // HAPUS: m_cullingFuture.wait() dan m_uboFuture.wait()
    
        m_commandBuffers[f].reset();
        recordCommandBuffer(m_commandBuffers[f], imageIndex);
    
        {
            TimelineSubmitInfo tsi;
            tsi.addWaitBinary(m_computeFinishedSemaphores[f],
                vk::PipelineStageFlagBits::eVertexShader);
            tsi.addWaitBinary(m_imageAvailableSemaphores[f],
                vk::PipelineStageFlagBits::eColorAttachmentOutput);
            tsi.commandBuffers.push_back(m_commandBuffers[f]);
            tsi.addSignalBinary(m_renderFinishedSemaphores[f]);
            if (!m_computeTimeline.isNull()) {
                uint64_t sigVal = m_computeTimeline.nextSignalValue();
                tsi.addSignalTimeline(m_computeTimeline.semaphore, sigVal);
            }
            m_graphicsQueue.submit(*tsi.build(), m_inFlightFences[f]);
        }
    
        {
            vk::PresentInfoKHR presentInfo{};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores    = &m_renderFinishedSemaphores[f];
            vk::SwapchainKHR swapChains[]  = {m_swapChain};
            presentInfo.swapchainCount     = 1;
            presentInfo.pSwapchains        = swapChains;
            presentInfo.pImageIndices      = &imageIndex;
            (void)m_presentQueue.presentKHR(presentInfo);
        }
    
        // Precompute frame N+1
        uint32_t nextFrame = (f + 1) % m_maxFrames;
        m_cullingJobHandle = JobHandle{};
        m_uboJobHandle     = JobHandle{};
    
        m_cullingJobHandle = m_jobSystem->submit([this, nextFrame]() {
            updateGameState(nextFrame);
        }, TaskPriority::Critical);
    
        m_currentFrame = nextFrame;
    }

    // ==========================================================================
    // RECORD COMMAND BUFFER
    // Menggunakan RenderGraph::EmitBarriers untuk automatic barrier management
    // Level geometry menggunakan m_mergedDrawCalls (FrustumCullingManager merged)
    // dengan fallback ke m_visibleTiles (tile-based) jika belum terisi
    // ==========================================================================
    void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {
        vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);
        m_gpuProfiler.beginFrame(commandBuffer, m_currentFrame);

        // ══ PASS 0-5: SHADOW AAA ══════════════════════════════════════════════
        //   0. Depth pre-pass   -> depth + oct-normal + velocity
        //   1. Cascade pass x4  -> shadowCascadeArray
        //   2. csm_resolve      -> PCSS blocker search + PCF
        //   3. shadow_temporal  -> reprojection + variance clipping
        //   4-5. bilateral blur H/V -> shadowFinal (dibaca main pass di binding 1)
        //
        // Sengaja TANPA frustum culling: pre-pass harus meliputi semua yang
        // terlihat kamera, dan cascade pass harus meliputi caster DI LUAR frustum
        // kamera (kalau di-cull, bayangan objek di belakang pemain akan hilang).
        // Culling per-cascade DIAKTIFKAN di bawah (lihat cabang cascadeIndex >= 0).
        {
            // cascadeIndex: -1 = depth pre-pass, 0..NUM_CASCADES-1 = pass cascade.
            auto drawScene = [this](vk::CommandBuffer cb, int cascadeIndex) {
                vk::DeviceSize off[] = {0};

                cb.bindVertexBuffers(0, 1, &m_vertexBuffer, off);
                cb.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint32);

                if (cascadeIndex < 0) {
                    // Pre-pass: seluruh level, TANPA culling kamera. Pre-pass
                    // memasok depth dan normal untuk setiap piksel layar; kalau
                    // ada yang hilang, csm_resolve akan salah merekonstruksi
                    // posisi dunia di situ.
                    cb.drawIndexed(m_levelIndexCount, 1, 0, 0, 0);
                } else if (!ShadowAAA::Cfg::CASCADE_CULLING) {
                    // Culling dimatikan: gambar seluruh level ke tiap cascade.
                    // Ini keadaan yang sama persis dengan v8, jadi berguna untuk
                    // memastikan apakah cullingnya yang bermasalah.
                    cb.drawIndexed(m_levelIndexCount, 1, 0, 0, 0);
                    m_shadowDrawCalls[static_cast<size_t>(cascadeIndex)] = 1;
                    m_shadowIndices[static_cast<size_t>(cascadeIndex)] = m_levelIndexCount;
                } else {
                    // ---- CULLING PER-CASCADE ------------------------------
                    // Cascade dekat cuma selebar beberapa meter, jadi mayoritas
                    // level jatuh di luarnya. Cascade jauh membentang ratusan
                    // meter dan nyaris tidak terbantu — dan memang seharusnya
                    // begitu: yang perlu dipangkas justru cascade dekat, karena
                    // ke sanalah resolusi paling padat dialokasikan.
                    //
                    // Tile yang lolos digabung bila rentang indeksnya
                    // bersambung, supaya tidak jadi puluhan draw call kecil.
                    uint32_t runFirst = 0, runCount = 0, drawn = 0;

                    uint32_t idxSent = 0;
                    auto flush = [&]() {
                        if (runCount) {
                            cb.drawIndexed(runCount, 1, runFirst, 0, 0);
                            idxSent += runCount;
                            runCount = 0;
                            ++drawn;
                        }
                    };

                    for (const auto& t : m_tiles) {
                        if (!m_shadowAAA.cascadeIntersectsAABB(
                                m_currentFrame, static_cast<uint32_t>(cascadeIndex),
                                t.aabb.min, t.aabb.max)) {
                            flush();
                            continue;
                        }
                        if (runCount && t.firstIndex == runFirst + runCount) {
                            runCount += t.indexCount;
                        } else {
                            flush();
                            runFirst  = t.firstIndex;
                            runCount  = t.indexCount;
                        }
                    }
                    flush();
                    m_shadowDrawCalls[static_cast<size_t>(cascadeIndex)] = drawn;
                    m_shadowIndices[static_cast<size_t>(cascadeIndex)]   = idxSent;
                }

                // Lantai selalu digambar: satu quad menutupi seluruh level,
                // jadi ia memotong setiap cascade.
                cb.bindVertexBuffers(0, 1, &m_floorVertexBuffer, off);
                cb.bindIndexBuffer(m_floorIndexBuffer, 0, vk::IndexType::eUint32);
                cb.drawIndexed(6, 1, 0, 0, 0);

                // Bola dinamis. Selalu digambar tanpa culling: ia satu objek
                // kecil yang posisinya berubah tiap frame, jadi AABB-nya tidak
                // ada di m_tiles dan tidak bisa ikut jalur culling statis.
                // v43: di cascade jauh bola cuma beberapa texel — bayangannya
                // sudah tidak berbentuk, tapi menggambarnya memaksa cascade
                // 2048^2 itu dirender ulang tiap frame. Pre-pass (cascadeIndex
                // < 0) dan main pass tetap menggambarnya utuh.
                const bool drawBall = Config::BALL_ENABLED && m_ballIndexCount > 0
                    && (cascadeIndex < 0
                        || m_shadowAAA.casterVisibleInCascade(cascadeIndex,
                                                              Config::BALL_RADIUS));
                if (drawBall) {
                    cb.bindVertexBuffers(0, 1, &m_ballVertexBuffer, off);
                    cb.bindIndexBuffer(m_ballIndexBuffer, 0, vk::IndexType::eUint32);
                    cb.drawIndexed(m_ballIndexCount, 1, 0, 0, 0);
                }
            };
            m_shadowAAA.record(commandBuffer, m_currentFrame, drawScene, &m_gpuProfiler);
        }

        vk::DeviceSize offsets[] = {0};

        // ── PASS 2: MAIN RENDER ───────────────────────────────────────────────
        std::array<vk::ClearValue,2> clearValues{};
        clearValues[0].color        = std::array<float,4>{0.1f, 0.1f, 0.15f, 1.0f};
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        vk::RenderPassBeginInfo mainRP{};
        mainRP.renderPass        = m_renderPass;
        mainRP.framebuffer       = m_hdrFramebuffer;   // offscreen, bukan swapchain
        mainRP.renderArea.offset = vk::Offset2D{0,0};
        mainRP.renderArea.extent = m_swapChainExtent;
        mainRP.clearValueCount   = 2;
        mainRP.pClearValues      = clearValues.data();

        commandBuffer.beginRenderPass(mainRP, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

        // Level geometry — pilih sumber: merged per-face atau tile-based
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            m_pipelineLayout, 0, 1, &m_descriptorSetsLevel[m_currentFrame], 0, nullptr);
        commandBuffer.bindVertexBuffers(0, 1, &m_vertexBuffer, offsets);
        commandBuffer.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint32);

        {
            std::lock_guard<std::mutex> lk(m_visibleMutex);

            if (!m_mergedDrawCalls.empty()) {
                // === Prioritas 1: Per-face merged DrawCalls (FrustumCullingManager) ===
                // Satu drawIndexed per batch merged, mengurangi API call overhead
                for (const auto& dc : m_mergedDrawCalls) {
                    commandBuffer.drawIndexed(
                        dc.indexCount, 1, dc.firstIndex, dc.vertexOffset, 0);
                }
            } else {
                // === Fallback: Tile-based CulledDraw ===
                for (const auto& draw : m_visibleTiles) {
                    commandBuffer.drawIndexed(draw.indexCount, 1, draw.firstIndex, 0, 0);
                }
            }
        }

        // Floor (selalu visible)
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            m_pipelineLayout, 0, 1, &m_descriptorSetsFloor[m_currentFrame], 0, nullptr);
        commandBuffer.bindVertexBuffers(0, 1, &m_floorVertexBuffer, offsets);
        commandBuffer.bindIndexBuffer(m_floorIndexBuffer, 0, vk::IndexType::eUint32);
        commandBuffer.drawIndexed(6, 1, 0, 0, 0);

        // Bola dinamis. Memakai descriptor set LEVEL, bukan lantai: parameter
        // materialnya sama dengan dinding, dan warnanya datang dari warna
        // verteks sehingga tidak perlu set sendiri.
        if (Config::BALL_ENABLED && m_ballIndexCount > 0) {
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                m_pipelineLayout, 0, 1, &m_descriptorSetsLevel[m_currentFrame], 0, nullptr);
            commandBuffer.bindVertexBuffers(0, 1, &m_ballVertexBuffer, offsets);
            commandBuffer.bindIndexBuffer(m_ballIndexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(m_ballIndexCount, 1, 0, 0, 0);
        }

        // ── LANGIT (v36) — digambar TERAKHIR di render pass ini ──────────────
        //
        // Urutannya penting dan sengaja: depth buffer sudah terisi geometri,
        // dan pipeline langit menguji LESS_OR_EQUAL terhadap gl_Position.z=1.0
        // tanpa menulis depth. Jadi fragmennya hanya lahir di piksel yang belum
        // ditulis siapa pun. Menggambarnya lebih dulu akan memberi hasil yang
        // sama tapi membayar overdraw satu layar penuh.
        //
        // Descriptor set LEVEL dipakai apa adanya: sky.frag cuma butuh binding 0
        // (UBO, untuk mvp / cameraPos / lightPos) dan binding 2 (buffer
        // volumetric). Keduanya identik di set level maupun floor.
        if (m_skyPipeline) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_skyPipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                m_pipelineLayout, 0, 1, &m_descriptorSetsLevel[m_currentFrame], 0, nullptr);

            glm::vec4 spc{Config::SKY_TURBIDITY,
                          Config::SKY_ZENITH_LUMA,
                          Config::SKY_SUN_INTENSITY,
                          Config::SKY_SHAFT_SCALE};
            commandBuffer.pushConstants(m_pipelineLayout,
                vk::ShaderStageFlagBits::eFragment, 0, sizeof(spc), &spc);

            // Tiga verteks tanpa vertex buffer — segitiga layar penuh.
            commandBuffer.draw(3, 1, 0, 0);
        }

        m_gpuProfiler.stamp(commandBuffer, m_currentFrame, ShadowAAA::GpuProfiler::MainPass);

        commandBuffer.endRenderPass();   // main pass selesai, HDR siap dibaca

        // ── PASS 2.9: TAA RESOLVE ────────────────────────────────────────────
        // Antara main pass dan komposit: membaca warna frame ini (ber-jitter),
        // history frame lalu, motion vector, dan depth; menulis history baru.
        // Komposit lalu men-tonemap history itu, bukan warna mentah.
        m_taaParity ^= 1u;
        {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_taaPipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                m_taaPipelineLayout, 0, 1, &m_taaSets[m_taaParity], 0, nullptr);

            // Blend dan penajaman ikut diskalakan: kalau jitter mati, tidak ada
            // yang perlu ditumpuk, dan unsharp cuma akan menajamkan aliasing.
            glm::vec4 tpc{Config::TAA_BLEND_MAX * m_taaHealth,
                          0.0f,   // ketajaman pindah ke composite.frag (di luar loop)
                          (Config::TAA_ENABLED && m_taaHealth > 0.01f) ? 1.0f : 0.0f,
                          // Sama seperti params4.x: 0,2 detik terlalu kecil untuk
                          // frame rate di rasterizer perangkat lunak, dan itu
                          // membuat batas waktu blend TAA tidak pernah menggigit.
                          glm::clamp(m_lastDeltaTime, 1.0f/240.0f, 4.0f)};
            commandBuffer.pushConstants(m_taaPipelineLayout,
                vk::ShaderStageFlagBits::eCompute, 0, sizeof(tpc), &tpc);

            commandBuffer.dispatch((m_swapChainExtent.width  + 7) / 8,
                                   (m_swapChainExtent.height + 7) / 8, 1);

            // Serahkan history yang baru ditulis ke fragment shader komposit.
            vk::MemoryBarrier mb{};
            mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
            mb.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eFragmentShader, {}, 1, &mb, 0, nullptr, 0, nullptr);
        }

        // ── PASS 3: KOMPOSIT — tonemap HDR ke swapchain ──────────────────────
        {
            vk::RenderPassBeginInfo cRP{};
            cRP.renderPass        = m_compositeRenderPass;
            cRP.framebuffer       = m_swapChainFramebuffers[imageIndex];
            cRP.renderArea.offset = vk::Offset2D{0,0};
            cRP.renderArea.extent = m_swapChainExtent;
            cRP.clearValueCount   = 0;      // loadOp eDontCare, seluruh layar ditimpa
            commandBuffer.beginRenderPass(cRP, vk::SubpassContents::eInline);

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_compositePipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                m_compositePipelineLayout, 0, 1, &m_compositeSets[m_taaParity], 0, nullptr);

            struct CompositePC { glm::vec4 params; glm::vec4 fxaa; } cpc{};
            cpc.params = glm::vec4(
                Config::TONEMAP_EXPOSURE,
                Config::TONEMAP_ACES ? 1.0f : 0.0f,
                m_swapchainIsSrgb ? 0.0f : 1.0f,
                // Penajaman dikerjakan DI SINI, bukan di taa_resolve —
                // di sini hasilnya tidak pernah ditulis kembali ke
                // history, jadi tidak bisa menumpuk antar frame.
                Config::TAA_ENABLED ? Config::TAA_SHARPNESS * m_taaHealth : 0.0f);
            cpc.fxaa = glm::vec4(
                Config::FXAA_ENABLED ? Config::FXAA_SUBPIX : -1.0f,
                Config::FXAA_EDGE_THRESHOLD,
                Config::FXAA_EDGE_THRESHOLD_MIN,
                Config::FXAA_DEBUG);
            commandBuffer.pushConstants(m_compositePipelineLayout,
                vk::ShaderStageFlagBits::eFragment, 0, sizeof(cpc), &cpc);

            // Tiga verteks tanpa vertex buffer — segitiga layar penuh.
            commandBuffer.draw(3, 1, 0, 0);
        }

        // ── PASS 3.5: JOYSTICK UI (setelah tonemap) ──────────────────────────
        if (m_joystickPipeline) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_joystickPipeline);
            commandBuffer.bindVertexBuffers(0, 1, &m_joystickVertexBuffer, offsets);

            vk::Viewport vp{};
            vp.x = 0; vp.y = 0;
            vp.width    = (float)m_swapChainExtent.width;
            vp.height   = (float)m_swapChainExtent.height;
            vp.minDepth = 0; vp.maxDepth = 1;
            commandBuffer.setViewport(0, 1, &vp);
            vk::Rect2D sc{}; sc.offset = vk::Offset2D{0,0}; sc.extent = m_swapChainExtent;
            commandBuffer.setScissor(0, 1, &sc);

            JoystickPC pcBase{};
            pcBase.offsetAndScale = glm::vec4(m_joystickCenter.x, m_joystickCenter.y,
                                              m_joystickRadius, 0.0f);
            pcBase.resolution     = glm::vec4((float)m_swapChainExtent.width,
                                              (float)m_swapChainExtent.height, 0, 0);
            pcBase.color          = glm::vec4(0.2f, 0.2f, 0.2f, 0.6f);
            commandBuffer.pushConstants(m_joystickPipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0, sizeof(JoystickPC), &pcBase);
            commandBuffer.draw(m_joystickVertexCount, 1, 0, 0);

            JoystickPC pcKnob{};
            pcKnob.offsetAndScale = glm::vec4(m_joystickKnobPos.x, m_joystickKnobPos.y,
                                              m_joystickKnobRadius, 0.0f);
            pcKnob.resolution = pcBase.resolution;
            float intensity   = (glm::length(m_moveInputVector) > Config::JOYSTICK_DEADZONE)
                                ? 1.0f : 0.8f;
            pcKnob.color      = glm::vec4(intensity, intensity, intensity, 0.9f);
            commandBuffer.pushConstants(m_joystickPipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0, sizeof(JoystickPC), &pcKnob);
            commandBuffer.draw(m_joystickVertexCount, 1, 0, 0);
        }

        commandBuffer.endRenderPass();
        m_gpuProfiler.stamp(commandBuffer, m_currentFrame, ShadowAAA::GpuProfiler::UiPass);
        commandBuffer.end();
    }

    // ==========================================================================
    // COLLISION
    // ==========================================================================
    bool checkCollision(const glm::vec3& pos) {
        int mH = (int)levelMap.size();
        int mW = (int)levelMap[0].size();
        float ox = -(mW * Config::MAP_SCALE) / 2.0f;
        float oz = -(mH * Config::MAP_SCALE) / 2.0f;

        std::array<glm::vec2,4> pts = {{
            {pos.x + Config::PLAYER_RADIUS, pos.z + Config::PLAYER_RADIUS},
            {pos.x - Config::PLAYER_RADIUS, pos.z + Config::PLAYER_RADIUS},
            {pos.x + Config::PLAYER_RADIUS, pos.z - Config::PLAYER_RADIUS},
            {pos.x - Config::PLAYER_RADIUS, pos.z - Config::PLAYER_RADIUS}
        }};
        for (auto& p : pts) {
            float fx = (p.x - (Config::MAP_SCALE/2.0f) - ox) / Config::MAP_SCALE;
            float fz = (p.y - (Config::MAP_SCALE/2.0f) - oz) / Config::MAP_SCALE;
            int mx = (int)round(fx), mz = (int)round(fz);
            if (mx >= 0 && mx < mW && mz >= 0 && mz < mH)
                if (levelMap[mz][mx] == 1) return true;
        }
        return false;
    }

    // ==========================================================================
    // JOYSTICK LAYOUT
    // ==========================================================================
    void updateJoystickLayout() {
        float w = (float)m_swapChainExtent.width;
        float h = (float)m_swapChainExtent.height;
        m_joystickRadius     = std::min(w,h) * Config::JOYSTICK_RELATIVE_SIZE;
        m_joystickKnobRadius = m_joystickRadius * Config::JOYSTICK_KNOB_RATIO;
        m_joystickCenter     = glm::vec2(Config::JOYSTICK_MARGIN + m_joystickRadius,
                                         h - Config::JOYSTICK_MARGIN - m_joystickRadius);
        m_joystickKnobPos    = m_joystickCenter;
    }

    // ==========================================================================
    // INPUT — thread-safe via m_inputMutex
    // ==========================================================================
    void handleInput(bool& running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
    
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
                return;
            }
    
            else if (event.type == SDL_EVENT_WINDOW_RESIZED) {
                m_swapChainExtent.width  = event.window.data1;
                m_swapChainExtent.height = event.window.data2;
                updateJoystickLayout();
                if (m_swapChainExtent.height > 0) {
                    std::lock_guard<std::mutex> lk(m_inputMutex);
                    m_camera.SetAspectRatio(
                        m_swapChainExtent.width / (float)m_swapChainExtent.height);
                }
            }
    
            // ── TOUCH DOWN ────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_FINGER_DOWN) {
                SDL_FingerID id = event.tfinger.fingerID;
                float tx = event.tfinger.x * m_swapChainExtent.width;
                float ty = event.tfinger.y * m_swapChainExtent.height;
                float dist = glm::length(glm::vec2(tx, ty) - m_joystickCenter);
    
                TouchPoint tp{};
                tp.fingerId = id;
                tp.position = {tx, ty};
                tp.isActive = true;
    
                if (dist < m_joystickRadius * 1.5f) {
                    tp.isJoystick = true;
                    std::lock_guard<std::mutex> lk(m_inputMutex);
                    m_joystickKnobPos = m_joystickCenter;
                    m_moveInputVector = glm::vec2(0.0f);
                } else {
                    tp.isJoystick = false;
                }
                m_activeTouches[id] = tp;
            }
    
            // ── TOUCH MOVE ────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_FINGER_MOTION) {
                SDL_FingerID id = event.tfinger.fingerID;
                if (!m_activeTouches.count(id)) continue;
    
                TouchPoint& tp = m_activeTouches[id];
                tp.position.x = event.tfinger.x * m_swapChainExtent.width;
                tp.position.y = event.tfinger.y * m_swapChainExtent.height;
    
                if (tp.isJoystick) {
                    glm::vec2 delta = tp.position - m_joystickCenter;
                    float dist = glm::length(delta);
                    std::lock_guard<std::mutex> lk(m_inputMutex);
                    if (dist > 5.0f) {
                        if (dist > m_joystickRadius) {
                            glm::vec2 clamped = glm::normalize(delta) * m_joystickRadius;
                            m_joystickKnobPos = m_joystickCenter + clamped;
                            m_moveInputVector = clamped / m_joystickRadius;
                        } else {
                            m_joystickKnobPos = m_joystickCenter + delta;
                            m_moveInputVector = delta / m_joystickRadius;
                        }
                    } else {
                        m_joystickKnobPos = m_joystickCenter;
                        m_moveInputVector = glm::vec2(0.0f);
                    }
                } else {
                    float dx = event.tfinger.dx * m_swapChainExtent.width;
                    float dy = event.tfinger.dy * m_swapChainExtent.height;
                    std::lock_guard<std::mutex> lk(m_inputMutex);
                    m_camera.Rotate(dx * 0.005f, dy * 0.005f);
                    m_cameraYaw         = m_camera.GetYaw();
                    m_cameraPitch       = m_camera.GetPitch();
                    m_cameraOrientation = m_camera.GetOrientation();
                }
            }
    
            // ── TOUCH UP ──────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_FINGER_UP) {
                SDL_FingerID id = event.tfinger.fingerID;
                if (m_activeTouches.count(id)) {
                    if (m_activeTouches[id].isJoystick) {
                        std::lock_guard<std::mutex> lk(m_inputMutex);
                        m_joystickKnobPos = m_joystickCenter;
                        m_moveInputVector = glm::vec2(0.0f);
                    }
                    m_activeTouches.erase(id);
                }
            }
    
            // ── MOUSE DOWN ────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    float mx   = (float)event.button.x;
                    float my   = (float)event.button.y;
                    float dist = glm::length(glm::vec2(mx, my) - m_joystickCenter);
    
                    SDL_FingerID fakeId = 999;
                    TouchPoint tp{};
                    tp.fingerId = fakeId;
                    tp.position = {mx, my};
                    tp.isActive = true;
    
                    if (dist < m_joystickRadius * 1.5f) {
                        tp.isJoystick = true;
                        std::lock_guard<std::mutex> lk(m_inputMutex);
                        m_joystickKnobPos = m_joystickCenter;
                        m_moveInputVector = glm::vec2(0.0f);
                    } else {
                        tp.isJoystick = false;
                    }
                    m_activeTouches[fakeId] = tp;
                }
            }
    
            // ── MOUSE MOVE ────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                SDL_FingerID fakeId = 999;
                if (!m_activeTouches.count(fakeId)) continue;
    
                TouchPoint& tp = m_activeTouches[fakeId];
                tp.position = {(float)event.motion.x, (float)event.motion.y};
    
                if (tp.isJoystick) {
                    glm::vec2 delta = tp.position - m_joystickCenter;
                    float dist = glm::length(delta);
                    std::lock_guard<std::mutex> lk(m_inputMutex);
                    if (dist > m_joystickRadius) {
                        glm::vec2 clamped = glm::normalize(delta) * m_joystickRadius;
                        m_joystickKnobPos = m_joystickCenter + clamped;
                        m_moveInputVector = clamped / m_joystickRadius;
                    } else if (dist > 5.0f) {
                        m_joystickKnobPos = m_joystickCenter + delta;
                        m_moveInputVector = delta / m_joystickRadius;
                    } else {
                        m_joystickKnobPos = m_joystickCenter;
                        m_moveInputVector = glm::vec2(0.0f);
                    }
                } else {
                    if (event.motion.state & SDL_BUTTON_LMASK) {
                        std::lock_guard<std::mutex> lk(m_inputMutex);
                        m_camera.Rotate(
                            event.motion.xrel * 0.002f,
                            event.motion.yrel * 0.002f);
                        m_cameraYaw         = m_camera.GetYaw();
                        m_cameraPitch       = m_camera.GetPitch();
                        m_cameraOrientation = m_camera.GetOrientation();
                    }
                }
            }
    
            // ── MOUSE UP ──────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                SDL_FingerID fakeId = 999;
                if (m_activeTouches.count(fakeId)) {
                    if (m_activeTouches[fakeId].isJoystick) {
                        std::lock_guard<std::mutex> lk(m_inputMutex);
                        m_joystickKnobPos = m_joystickCenter;
                        m_moveInputVector = glm::vec2(0.0f);
                    }
                    m_activeTouches.erase(fakeId);
                }
            }
    
            // ── KEYBOARD ──────────────────────────────────────────────────────
            else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE) {
                    running = false;
                    return;
                }
            }
        }
    }

    // ==========================================================================
    // VULKAN INSTANCE & DEVICE SETUP
    // ==========================================================================
    bool checkValidationLayerSupport() {
        auto layers = vk::enumerateInstanceLayerProperties();
        for (const char* name : validationLayers) {
            bool found = false;
            for (auto& lp : layers) if (strcmp(name, lp.layerName) == 0) { found = true; break; }
            if (!found) return false;
        }
        return true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT sev,
        vk::DebugUtilsMessageTypeFlagsEXT,
        const vk::DebugUtilsMessengerCallbackDataEXT* cb, void*) {
        if (sev >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            // v74: ke logcat juga. Callback ini cuma hidup kalau
            // ENABLE_VALIDATION true, jadi di Android ia tidak akan pernah
            // dipanggil — tapi membiarkan std::cerr di sini berarti kalau
            // suatu saat kamu uji di perangkat yang PUNYA layer validasi,
            // pesannya hilang tanpa jejak.
            SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "[VK] %s", cb->pMessage);
        return VK_FALSE;
    }

    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& ci) {
        ci.messageSeverity =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        ci.messageType =
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        ci.pfnUserCallback = debugCallback;
    }

    void createInstance() {
        if (Config::ENABLE_VALIDATION && !checkValidationLayerSupport())
            throw std::runtime_error("Validation layers requested, but not available");

        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName   = Config::APP_NAME.c_str();
        appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
        appInfo.pEngineName        = Config::ENGINE_NAME.c_str();
        appInfo.engineVersion      = VK_MAKE_VERSION(1,0,0);
        // v72: 1.3 -> 1.1.
        //
        // vulkaninfo pada perangkat ini melaporkan apiVersion 1.1.131. Meminta
        // 1.3 memang tidak selalu langsung gagal — loader mengizinkannya — tapi
        // ia membuat seluruh kode di bawahnya BOLEH mengira fitur inti 1.2/1.3
        // tersedia, padahal tidak. Lebih baik jujur sejak awal.
        appInfo.apiVersion         = VK_API_VERSION_1_1;

        uint32_t sdlCount = 0;
        const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlCount);
        std::vector<const char*> extensions(sdlExts, sdlExts + sdlCount);
        if (Config::ENABLE_VALIDATION)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        vk::InstanceCreateInfo ci{};
        ci.pApplicationInfo        = &appInfo;
        ci.enabledExtensionCount   = (uint32_t)extensions.size();
        ci.ppEnabledExtensionNames = extensions.data();

        vk::DebugUtilsMessengerCreateInfoEXT debugCI;
        if (Config::ENABLE_VALIDATION) {
            populateDebugMessengerCreateInfo(debugCI);
            ci.enabledLayerCount   = (uint32_t)validationLayers.size();
            ci.ppEnabledLayerNames = validationLayers.data();
            ci.pNext               = &debugCI;
        }
        m_instance = vk::createInstance(ci);
    }

    void setupDebugMessenger() {
        if (!Config::ENABLE_VALIDATION) return;
        vk::DebugUtilsMessengerCreateInfoEXT ci;
        populateDebugMessengerCreateInfo(ci);
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
            m_instance.getProcAddr("vkCreateDebugUtilsMessengerEXT");
        if (!fn) throw std::runtime_error("Cannot load vkCreateDebugUtilsMessengerEXT");
        fn(m_instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&ci),
           nullptr, reinterpret_cast<VkDebugUtilsMessengerEXT*>(&m_debugMessenger));
    }

    void createSurface() {
        VkSurfaceKHR s;
        if (!SDL_Vulkan_CreateSurface(window, m_instance, nullptr, &s))
            throw std::runtime_error("Surface creation failed");
        m_surface = s;
    }

    // Queue family selection: graphics, present, dan compute (untuk async compute)
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        std::optional<uint32_t> computeFamily; // dedicated async compute
        bool isComplete() const {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice dev) {
        QueueFamilyIndices idx;
        auto qf = dev.getQueueFamilyProperties();
        for (uint32_t i = 0; i < (uint32_t)qf.size(); i++) {
            if (qf[i].queueFlags & vk::QueueFlagBits::eGraphics)
                idx.graphicsFamily = i;
            if (dev.getSurfaceSupportKHR(i, m_surface))
                idx.presentFamily = i;
            // Dedicated compute: has compute but NOT graphics
            if ((qf[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                !(qf[i].queueFlags & vk::QueueFlagBits::eGraphics))
                idx.computeFamily = i;
            if (idx.isComplete()) break;
        }
        return idx;
    }

    bool isDeviceSuitable(vk::PhysicalDevice dev) {
        auto idx = findQueueFamilies(dev);
        bool extOK = checkDeviceExtensionSupport(dev);
        bool scOK  = false;
        if (extOK) {
            auto sc = dev.getSurfaceFormatsKHR(m_surface);
            auto pm = dev.getSurfacePresentModesKHR(m_surface);
            scOK = !sc.empty() && !pm.empty();
        }
        return idx.isComplete() && extOK && scOK;
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice dev) {
        auto exts = dev.enumerateDeviceExtensionProperties();
        std::set<std::string> req = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (auto& e : exts) req.erase(e.extensionName);
        return req.empty();
    }

    void pickPhysicalDevice() {
        auto devs = m_instance.enumeratePhysicalDevices();
        for (auto& d : devs) if (isDeviceSuitable(d)) { m_physicalDevice = d; break; }
        if (!m_physicalDevice) throw std::runtime_error("No suitable GPU found");
    }

    void createLogicalDevice() {
        auto idx = findQueueFamilies(m_physicalDevice);
        m_graphicsFamily = idx.graphicsFamily.value();
        m_presentFamily  = idx.presentFamily.value();

        std::set<uint32_t> uniqueFamilies = {m_graphicsFamily, m_presentFamily};
        if (idx.computeFamily.has_value()) {
            m_computeFamily  = idx.computeFamily.value();
            m_hasAsyncCompute = true;
            uniqueFamilies.insert(m_computeFamily);
        } else {
            // Fallback: share graphics queue untuk compute
            m_computeFamily   = m_graphicsFamily;
            m_hasAsyncCompute = false;
        }

        float qPriority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> qcis;
        for (uint32_t qf : uniqueFamilies) {
            vk::DeviceQueueCreateInfo qi{};
            qi.queueFamilyIndex = qf;
            qi.queueCount       = 1;
            qi.pQueuePriorities = &qPriority;
            qcis.push_back(qi);
        }

        vk::PhysicalDeviceFeatures features{};
        const char* exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        // FIX: Enable timelineSemaphore jika device support, agar vkCreateSemaphore
        // dengan VK_SEMAPHORE_TYPE_TIMELINE tidak ditolak validation layer.
        vk::PhysicalDeviceVulkan12Features vk12Features{};
        {
            vk::PhysicalDeviceFeatures2 f2{};
            f2.pNext = &vk12Features;
            m_physicalDevice.getFeatures2(&f2);
            // hanya aktifkan yang kita tahu tersedia
            if (!vk12Features.timelineSemaphore)
                vk12Features.timelineSemaphore = VK_FALSE;
        }

        vk::DeviceCreateInfo ci{};
        ci.queueCreateInfoCount    = (uint32_t)qcis.size();
        ci.pQueueCreateInfos       = qcis.data();
        ci.pEnabledFeatures        = &features;
        ci.enabledExtensionCount   = 1;
        ci.ppEnabledExtensionNames = exts;
        ci.pNext                   = &vk12Features; // chain Vulkan 1.2 features

        m_device = m_physicalDevice.createDevice(ci);
        m_graphicsQueue = m_device.getQueue(m_graphicsFamily, 0);
        m_presentQueue  = m_device.getQueue(m_presentFamily,  0);
        m_computeQueue  = m_device.getQueue(m_computeFamily,  0);
    }

    // ==========================================================================
    // SWAPCHAIN
    // ==========================================================================
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR    capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR>   presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice dev) {
        return {
            dev.getSurfaceCapabilitiesKHR(m_surface),
            dev.getSurfaceFormatsKHR(m_surface),
            dev.getSurfacePresentModesKHR(m_surface)
        };
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& f) {
        for (auto& fmt : f)
            if (fmt.format == vk::Format::eB8G8R8A8Srgb &&
                fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) return fmt;
        return f[0];
    }
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& pm) {
        // v50: ENABLE_VSYNC dulu dideklarasikan tapi TIDAK PERNAH dibaca —
        // present mode-nya dipilih tanpa melihatnya sama sekali. Sakelar yang
        // tidak tersambung ke apa pun lebih buruk daripada tidak ada sakelar:
        // ia membuat orang mengira sudah mencoba sesuatu.
        //
        //   true  -> FIFO / FIFO_RELAXED : menunggu vblank, tanpa tearing
        //   false -> MAILBOX / IMMEDIATE : frame terbaru menang, bisa tearing,
        //            tapi frame time apa adanya — yang diinginkan saat mengukur
        if (Config::ENABLE_VSYNC) {
            for (auto& m : pm) if (m == vk::PresentModeKHR::eFifoRelaxed) return m;
            return vk::PresentModeKHR::eFifo;          // dijamin selalu ada
        }
        for (auto& m : pm) if (m == vk::PresentModeKHR::eMailbox)   return m;
        for (auto& m : pm) if (m == vk::PresentModeKHR::eImmediate) return m;
        return vk::PresentModeKHR::eFifo;
    }
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& cap) {
        if (cap.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return cap.currentExtent;
        int w,h; SDL_GetWindowSize(window, &w, &h);
        return {
            std::clamp((uint32_t)w, cap.minImageExtent.width, cap.maxImageExtent.width),
            std::clamp((uint32_t)h, cap.minImageExtent.height, cap.maxImageExtent.height)
        };
    }

    void createSwapChain() {
        auto sc   = querySwapChainSupport(m_physicalDevice);
        auto fmt  = chooseSwapSurfaceFormat(sc.formats);
        auto pm   = chooseSwapPresentMode(sc.presentModes);
        auto ext  = chooseSwapExtent(sc.capabilities);
        uint32_t imgCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0 && imgCount > sc.capabilities.maxImageCount)
            imgCount = sc.capabilities.maxImageCount;

        vk::SwapchainCreateInfoKHR ci{};
        ci.surface          = m_surface;
        ci.minImageCount    = imgCount;
        ci.imageFormat      = fmt.format;
        ci.imageColorSpace  = fmt.colorSpace;
        ci.imageExtent      = ext;
        ci.imageArrayLayers = 1;
        ci.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment;
        ci.preTransform     = sc.capabilities.currentTransform;
        ci.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        ci.presentMode      = pm;
        LOG_INFO("Swapchain", std::string("present mode: ") + vk::to_string(pm)
                 + (Config::ENABLE_VSYNC ? "  (ENABLE_VSYNC = true)"
                                         : "  (ENABLE_VSYNC = false)"));
        ci.clipped          = VK_TRUE;

        uint32_t qfis[] = {m_graphicsFamily, m_presentFamily};
        if (m_graphicsFamily != m_presentFamily) {
            ci.imageSharingMode      = vk::SharingMode::eConcurrent;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices   = qfis;
        } else {
            ci.imageSharingMode = vk::SharingMode::eExclusive;
        }

        m_swapChain            = m_device.createSwapchainKHR(ci);
        m_swapChainImages      = m_device.getSwapchainImagesKHR(m_swapChain);
        m_swapChainImageFormat = fmt.format;
        m_swapChainExtent      = ext;
    }

    // ==========================================================================
    // IMAGE VIEWS & DEPTH & SHADOW
    // ==========================================================================
    vk::Format findSupportedFormat(const std::vector<vk::Format>& cands,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features) {
        for (auto f : cands) {
            auto p = m_physicalDevice.getFormatProperties(f);
            if (tiling == vk::ImageTiling::eLinear  && (p.linearTilingFeatures  & features) == features) return f;
            if (tiling == vk::ImageTiling::eOptimal && (p.optimalTilingFeatures & features) == features) return f;
        }
        throw std::runtime_error("No supported depth format");
    }
    vk::Format findDepthFormat() {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags props) {
        auto mp = m_physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((typeFilter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("No suitable memory type");
    }

    void createImageViews() {
        m_swapChainImageViews.resize(m_swapChainImages.size());
        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            vk::ImageViewCreateInfo ci{};
            ci.image    = m_swapChainImages[i];
            ci.viewType = vk::ImageViewType::e2D;
            ci.format   = m_swapChainImageFormat;
            ci.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            m_swapChainImageViews[i] = m_device.createImageView(ci);
        }
    }

    void createDepthResources() {
        vk::Format df = findDepthFormat();
        vk::ImageCreateInfo ii{};
        ii.imageType   = vk::ImageType::e2D;
        ii.extent      = vk::Extent3D{m_swapChainExtent.width, m_swapChainExtent.height, 1};
        ii.mipLevels   = 1; ii.arrayLayers = 1;
        ii.format      = df;
        ii.tiling      = vk::ImageTiling::eOptimal;
        ii.initialLayout = vk::ImageLayout::eUndefined;
        ii.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        ii.samples     = vk::SampleCountFlagBits::e1;
        ii.sharingMode = vk::SharingMode::eExclusive;
        m_depthImage = m_device.createImage(ii);

        auto mr = m_device.getImageMemoryRequirements(m_depthImage);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        m_depthImageMemory = m_device.allocateMemory(ai);
        m_device.bindImageMemory(m_depthImage, m_depthImageMemory, 0);

        vk::ImageViewCreateInfo vi{};
        vi.image    = m_depthImage;
        vi.viewType = vk::ImageViewType::e2D;
        vi.format   = df;
        vi.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
        m_depthImageView = m_device.createImageView(vi);
    }

    // createShadowResources / createShadowRenderPass / createShadowFramebuffer
    // DIHAPUS — digantikan ShadowAAA::Pipeline (cascade array + render pass sendiri).

    vk::ShaderModule createShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo ci{};
        ci.codeSize = code.size();
        ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        return m_device.createShaderModule(ci);
    }

    // createShadowPipeline DIHAPUS — cascade pipeline dibuat di ShadowAAA::Pipeline.

    void createRenderPass() {
        vk::AttachmentDescription ca{};
        // rgba16f, bukan format swapchain.
        //
        // Swapchain 8 bit per kanal memotong keras di 1.0, jadi setiap nilai di
        // atas satu — sorotan matahari, berkas volumetric yang menumpuk — hilang
        // sebelum sempat dipakai. Di sini rentangnya dipertahankan, dan pass
        // komposit yang memutuskan bagaimana memampatkannya ke layar.
        ca.format = vk::Format::eR16G16B16A16Sfloat;
        ca.samples= vk::SampleCountFlagBits::e1;
        ca.loadOp = vk::AttachmentLoadOp::eClear;
        ca.storeOp= vk::AttachmentStoreOp::eStore;
        ca.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        ca.stencilStoreOp= vk::AttachmentStoreOp::eDontCare;
        ca.initialLayout = vk::ImageLayout::eUndefined;
        // Bukan lagi ePresentSrcKHR: hasilnya dibaca pass komposit sebagai
        // texture, bukan langsung ditampilkan.
        ca.finalLayout   = vk::ImageLayout::eShaderReadOnlyOptimal;
        vk::AttachmentReference car{0, vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentDescription da{};
        da.format = findDepthFormat();
        da.samples= vk::SampleCountFlagBits::e1;
        da.loadOp = vk::AttachmentLoadOp::eClear;
        da.storeOp= vk::AttachmentStoreOp::eDontCare;
        da.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        da.stencilStoreOp= vk::AttachmentStoreOp::eDontCare;
        da.initialLayout = vk::ImageLayout::eUndefined;
        da.finalLayout   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        vk::AttachmentReference dar{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::SubpassDescription sp{};
        sp.pipelineBindPoint      = vk::PipelineBindPoint::eGraphics;
        sp.colorAttachmentCount   = 1; sp.pColorAttachments       = &car;
        sp.pDepthStencilAttachment = &dar;

        vk::SubpassDependency dep{};
        dep.srcSubpass   = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        // eComputeShader + eShaderRead WAJIB ada di sisi src.
        //
        // Target HDR cuma SATU image, dipakai bergantian oleh semua frame in
        // flight, dan pass TAA RESOLVE membacanya sebagai texture di tahap
        // COMPUTE. Tanpa dependency ini, main pass frame berikutnya boleh mulai
        // membersihkan dan menimpa HDR sementara TAA frame sebelumnya masih
        // membacanya — write-after-read yang tidak tersinkronisasi.
        //
        // m_inFlightFences tidak menutupinya: fence yang ditunggu milik SLOT
        // yang sama, yaitu dua sampai tiga frame ke belakang, bukan frame tepat
        // sebelumnya.
        dep.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                           vk::PipelineStageFlagBits::eEarlyFragmentTests   |
                           vk::PipelineStageFlagBits::eComputeShader;
        dep.srcAccessMask= vk::AccessFlagBits::eShaderRead;
        dep.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                           vk::PipelineStageFlagBits::eEarlyFragmentTests;
        dep.dstAccessMask= vk::AccessFlagBits::eColorAttachmentWrite |
                           vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        std::array<vk::AttachmentDescription,2> attachments = {ca, da};
        vk::RenderPassCreateInfo ci{};
        ci.attachmentCount = 2; ci.pAttachments = attachments.data();
        ci.subpassCount    = 1; ci.pSubpasses   = &sp;
        ci.dependencyCount = 1; ci.pDependencies= &dep;
        m_renderPass = m_device.createRenderPass(ci);
    }

    // Target warna HDR yang ditulis main pass.
    void createHdrTarget() {
        vk::ImageCreateInfo ii{};
        ii.imageType   = vk::ImageType::e2D;
        ii.extent      = vk::Extent3D{m_swapChainExtent.width, m_swapChainExtent.height, 1};
        ii.mipLevels   = 1; ii.arrayLayers = 1;
        ii.format      = vk::Format::eR16G16B16A16Sfloat;
        ii.tiling      = vk::ImageTiling::eOptimal;
        ii.initialLayout = vk::ImageLayout::eUndefined;
        ii.usage       = vk::ImageUsageFlagBits::eColorAttachment |
                         vk::ImageUsageFlagBits::eSampled;
        ii.samples     = vk::SampleCountFlagBits::e1;
        ii.sharingMode = vk::SharingMode::eExclusive;
        m_hdrImage = m_device.createImage(ii);

        auto mr = m_device.getImageMemoryRequirements(m_hdrImage);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                            vk::MemoryPropertyFlagBits::eDeviceLocal);
        m_hdrMemory = m_device.allocateMemory(ai);
        m_device.bindImageMemory(m_hdrImage, m_hdrMemory, 0);

        vk::ImageViewCreateInfo vi{};
        vi.image    = m_hdrImage;
        vi.viewType = vk::ImageViewType::e2D;
        vi.format   = vk::Format::eR16G16B16A16Sfloat;
        vi.subresourceRange = vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        m_hdrImageView = m_device.createImageView(vi);

        m_swapchainIsSrgb = (m_swapChainImageFormat == vk::Format::eB8G8R8A8Srgb ||
                             m_swapChainImageFormat == vk::Format::eR8G8B8A8Srgb);
        LOG_INFO("Composite", m_swapchainIsSrgb
            ? "Swapchain sRGB: gamma ditangani perangkat keras"
            : "Swapchain BUKAN sRGB: gamma dikerjakan di composite.frag");
    }

    // Pass komposit: membaca HDR, tonemap, tulis ke swapchain. UI digambar di
    // sini juga supaya tidak ikut kena tonemap.
    void createCompositeRenderPass() {
        vk::AttachmentDescription ca{};
        ca.format  = m_swapChainImageFormat;
        ca.samples = vk::SampleCountFlagBits::e1;
        // eDontCare, bukan eClear: segitiga layar penuh menimpa setiap piksel,
        // jadi membersihkannya lebih dulu adalah tulisan yang terbuang.
        ca.loadOp  = vk::AttachmentLoadOp::eDontCare;
        ca.storeOp = vk::AttachmentStoreOp::eStore;
        ca.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
        ca.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        ca.initialLayout  = vk::ImageLayout::eUndefined;
        ca.finalLayout    = vk::ImageLayout::ePresentSrcKHR;
        vk::AttachmentReference car{0, vk::ImageLayout::eColorAttachmentOptimal};

        vk::SubpassDescription sp{};
        sp.pipelineBindPoint    = vk::PipelineBindPoint::eGraphics;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments    = &car;

        // Menunggu main pass selesai MENULIS HDR sebelum pass ini MEMBACANYA.
        vk::SubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dep.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        dep.dstStageMask  = vk::PipelineStageFlagBits::eFragmentShader;
        dep.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        vk::RenderPassCreateInfo ci{};
        ci.attachmentCount = 1; ci.pAttachments  = &ca;
        ci.subpassCount    = 1; ci.pSubpasses    = &sp;
        ci.dependencyCount = 1; ci.pDependencies = &dep;
        m_compositeRenderPass = m_device.createRenderPass(ci);
    }

    void createGraphicsPipeline() {
        auto vc = readFile("main.vert.spv");
        auto fc = readFile("main.frag.spv");
        auto vm = createShaderModule(vc);
        auto fm = createShaderModule(fc);

        vk::PipelineShaderStageCreateInfo stages[2];
        stages[0] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eVertex,   vm, "main"};
        stages[1] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eFragment, fm, "main"};

        auto bind  = Vertex::getBindingDescription();
        auto attrs = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vi{};
        vi.vertexBindingDescriptionCount   = 1;  vi.pVertexBindingDescriptions      = &bind;
        // Diturunkan dari array, BUKAN ditulis tangan. Angka 3 yang dulu ada di
        // sini akan diam-diam membuang atribut keempat saat prevPos ditambahkan:
        // pipeline tetap dibuat, shader tetap terkompilasi, dan inPrevPos cuma
        // berisi nol. Motion vector jadi salah tanpa satu pun pesan error.
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions    = attrs.data();

        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.topology = vk::PrimitiveTopology::eTriangleList;

        vk::Viewport vp{0,0,(float)m_swapChainExtent.width,(float)m_swapChainExtent.height,0,1};
        vk::Rect2D sc{{0,0},m_swapChainExtent};
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount=1; vs.pViewports=&vp; vs.scissorCount=1; vs.pScissors=&sc;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode = vk::PolygonMode::eFill; rz.lineWidth = 1.0f;
        rz.cullMode    = vk::CullModeFlagBits::eNone;
        rz.frontFace   = vk::FrontFace::eCounterClockwise;

        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineDepthStencilStateCreateInfo ds{};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = vk::CompareOp::eLess;

        vk::PipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        cba.blendEnable = VK_FALSE;
        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        // Descriptor set layout
        vk::DescriptorSetLayoutBinding ubo{0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment};
        vk::DescriptorSetLayoutBinding smp{1, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment};
        // binding 2 = light shaft (setengah resolusi, dibesarkan sampler linear).
        // WAJIB ada di sini: main.frag mendeklarasikannya, dan Vulkan menolak
        // pipeline yang shader-nya memakai binding yang tidak ada di layout.
        vk::DescriptorSetLayoutBinding vol{2, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment};
        std::array<vk::DescriptorSetLayoutBinding,3> bindings = {ubo, smp, vol};
        vk::DescriptorSetLayoutCreateInfo dli{};
        dli.bindingCount = 3; dli.pBindings = bindings.data();
        m_descriptorSetLayout = m_device.createDescriptorSetLayout(dli);

        // Push constant untuk pipeline LANGIT (v36). Ia menumpang layout yang
        // sama dengan geometri supaya tidak perlu descriptor set kedua.
        // main.vert/main.frag tidak mendeklarasikannya, dan itu sah: sebuah
        // pipeline layout boleh punya rentang push constant yang tidak dipakai
        // semua shader yang memakainya. Nilainya cuma di-push tepat sebelum
        // draw langit.
        vk::PushConstantRange skyPC{};
        skyPC.stageFlags = vk::ShaderStageFlagBits::eFragment;
        skyPC.offset     = 0;
        skyPC.size       = sizeof(float) * 4;

        vk::PipelineLayoutCreateInfo pli{};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_descriptorSetLayout;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &skyPC;
        m_pipelineLayout = m_device.createPipelineLayout(pli);

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount=2; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia;
        pi.pViewportState=&vs; pi.pRasterizationState=&rz;
        pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.layout=m_pipelineLayout;
        pi.renderPass=m_renderPass;

        m_graphicsPipeline = m_device.createGraphicsPipeline(VK_NULL_HANDLE, pi).value;
        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);
    }

    // ==========================================================================
    // PIPELINE LANGIT (v36)
    //
    //   Segitiga layar penuh di dalam render pass UTAMA, digambar SESUDAH
    //   geometri. Tiga keputusan yang saling mengunci:
    //
    //   [1] depthTest NYALA, depthWrite MATI, compareOp LESS_OR_EQUAL, dan
    //       sky.vert menulis gl_Position.z = 1.0. Depth buffer di-clear ke 1.0,
    //       geometri menulis nilai < 1.0. Jadi 1.0 <= 1.0 lolos di piksel
    //       langit dan 1.0 <= 0.5 gagal di piksel geometri: nol overdraw, dan
    //       langit terhalang bangunan tanpa satu pun uji di fragment shader.
    //       LESS_OR_EQUAL dipilih, bukan EQUAL, supaya tidak bergantung pada
    //       kesetaraan float yang persis.
    //
    //   [2] Vertex input KOSONG. Verteksnya dibangun dari gl_VertexIndex, jadi
    //       vertex buffer yang kebetulan masih ter-bind dari draw sebelumnya
    //       tidak berpengaruh apa-apa.
    //
    //   [3] Memakai m_pipelineLayout dan m_renderPass yang sudah ada. sky.frag
    //       membaca binding 0 (UBO) dan binding 2 (buffer volumetric) dari
    //       descriptor set yang sama persis dengan geometri — jadi tidak ada
    //       descriptor set layout, pool, maupun image baru di build ini.
    //
    //   Kalau sky.vert.spv / sky.frag.spv belum ada (compile_shaders.sh belum
    //   diperbarui), fungsi ini TIDAK melempar exception. Ia memberi peringatan
    //   dan membiarkan m_skyPipeline null; draw-nya dilewati dan gambarnya
    //   kembali ke clear color seperti build sebelumnya.
    // ==========================================================================
    void createSkyPipeline() {
        if (!Config::SKY_ENABLED) {
            LOG_INFO("Sky", "SKY_ENABLED = false, pass langit dilewati");
            return;
        }

        auto vc = ShaderModule::ReadFile("sky.vert.spv");
        auto fc = ShaderModule::ReadFile("sky.frag.spv");
        if (vc.empty() || fc.empty()) {
            LOG_WARN("Sky", "sky.vert.spv / sky.frag.spv tidak ditemukan — "
                            "tambahkan keduanya ke compile_shaders.sh. "
                            "Pass langit dimatikan untuk run ini.");
            return;
        }

        auto vm = createShaderModule(vc);
        auto fm = createShaderModule(fc);

        vk::PipelineShaderStageCreateInfo stages[2];
        stages[0] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eVertex,   vm, "main"};
        stages[1] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eFragment, fm, "main"};

        // [2] tidak ada binding maupun atribut verteks.
        vk::PipelineVertexInputStateCreateInfo vi{};
        vi.vertexBindingDescriptionCount   = 0;
        vi.vertexAttributeDescriptionCount = 0;

        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.topology = vk::PrimitiveTopology::eTriangleList;

        vk::Viewport vp{0,0,(float)m_swapChainExtent.width,(float)m_swapChainExtent.height,0,1};
        vk::Rect2D sc{{0,0},m_swapChainExtent};
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount=1; vs.pViewports=&vp; vs.scissorCount=1; vs.pScissors=&sc;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode = vk::PolygonMode::eFill; rz.lineWidth = 1.0f;
        rz.cullMode    = vk::CullModeFlagBits::eNone;
        rz.frontFace   = vk::FrontFace::eCounterClockwise;

        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

        // [1] baca, jangan tulis.
        vk::PipelineDepthStencilStateCreateInfo ds{};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp   = vk::CompareOp::eLessOrEqual;

        vk::PipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        cba.blendEnable = VK_FALSE;
        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount=2; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia;
        pi.pViewportState=&vs; pi.pRasterizationState=&rz;
        pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.layout=m_pipelineLayout;
        pi.renderPass=m_renderPass;

        m_skyPipeline = m_device.createGraphicsPipeline(VK_NULL_HANDLE, pi).value;
        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);

        LOG_INFO("Sky", "Langit Preetham aktif — turbidity "
                        + std::to_string(Config::SKY_TURBIDITY)
                        + ", luminansi zenit " + std::to_string(Config::SKY_ZENITH_LUMA));
    }

    void createJoystickPipeline() {
        auto vc = readFile("joystick.vert.spv");
        auto fc = readFile("joystick.frag.spv");
        if (vc.empty() || fc.empty()) {
            LOG_WARN("App", "Joystick shaders not found — UI disabled");
            return;
        }
        auto vm = createShaderModule(vc);
        auto fm = createShaderModule(fc);

        vk::PipelineShaderStageCreateInfo stages[2];
        stages[0] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eVertex,   vm, "main"};
        stages[1] = {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eFragment, fm, "main"};

        auto bind  = JoystickVertex::getBindingDescription();
        auto attrs = JoystickVertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vi{};
        vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
        vi.vertexAttributeDescriptionCount=1; vi.pVertexAttributeDescriptions=attrs.data();

        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.topology = vk::PrimitiveTopology::eTriangleFan;

        std::array<vk::DynamicState,2> dyn = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynInfo{};
        dynInfo.dynamicStateCount=2; dynInfo.pDynamicStates=dyn.data();

        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount=1; vs.scissorCount=1;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode=vk::PolygonMode::eFill; rz.lineWidth=1.0f;
        rz.cullMode=vk::CullModeFlagBits::eNone;
        rz.frontFace=vk::FrontFace::eCounterClockwise;

        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineDepthStencilStateCreateInfo ds{};
        ds.depthTestEnable=VK_FALSE; ds.depthWriteEnable=VK_FALSE;
        ds.depthCompareOp=vk::CompareOp::eAlways;

        vk::PipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        cba.blendEnable=VK_TRUE;
        cba.srcColorBlendFactor=vk::BlendFactor::eSrcAlpha;
        cba.dstColorBlendFactor=vk::BlendFactor::eOneMinusSrcAlpha;
        cba.colorBlendOp=vk::BlendOp::eAdd;
        cba.srcAlphaBlendFactor=vk::BlendFactor::eOne;
        cba.dstAlphaBlendFactor=vk::BlendFactor::eZero;
        cba.alphaBlendOp=vk::BlendOp::eAdd;
        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount=1; cb.pAttachments=&cba;

        vk::PushConstantRange pcr{
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(JoystickPC)};
        vk::PipelineLayoutCreateInfo pli{};
        pli.setLayoutCount=0; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr;
        m_joystickPipelineLayout = m_device.createPipelineLayout(pli);

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount=2; pi.pStages=stages;
        pi.pVertexInputState=&vi; pi.pInputAssemblyState=&ia;
        pi.pViewportState=&vs; pi.pRasterizationState=&rz;
        pi.pMultisampleState=&ms; pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb; pi.pDynamicState=&dynInfo;
        // UI digambar di pass KOMPOSIT, setelah tonemap. Kalau ia tetap di main
        // pass, warnanya ikut dimampatkan kurva ACES dan joystick jadi kusam —
        // padahal UI memang dimaksudkan tampil apa adanya.
        pi.layout=m_joystickPipelineLayout; pi.renderPass=m_compositeRenderPass;

        m_joystickPipeline = m_device.createGraphicsPipeline(VK_NULL_HANDLE, pi).value;
        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);
    }

    // Dua history warna + pipeline resolve TAA.
    void createTaaResources() {
        for (int i = 0; i < 2; ++i) {
            vk::ImageCreateInfo ii{};
            ii.imageType = vk::ImageType::e2D;
            ii.extent    = vk::Extent3D{m_swapChainExtent.width, m_swapChainExtent.height, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1;
            ii.format    = vk::Format::eR16G16B16A16Sfloat;
            ii.tiling    = vk::ImageTiling::eOptimal;
            ii.initialLayout = vk::ImageLayout::eUndefined;
            // eTransferDst WAJIB: kedua image ini dinolkan dengan
            // vkCmdClearColorImage tepat di bawah, dan clear termasuk operasi
            // transfer. Ini kesalahan yang sama persis dengan image storage
            // ShadowAAA dulu — saya mengulanginya di sini.
            ii.usage     = vk::ImageUsageFlagBits::eStorage |
                           vk::ImageUsageFlagBits::eSampled |
                           vk::ImageUsageFlagBits::eTransferDst;
            ii.samples   = vk::SampleCountFlagBits::e1;
            ii.sharingMode = vk::SharingMode::eExclusive;
            m_taaImage[i] = m_device.createImage(ii);

            auto mr = m_device.getImageMemoryRequirements(m_taaImage[i]);
            vk::MemoryAllocateInfo ai{};
            ai.allocationSize  = mr.size;
            ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal);
            m_taaMemory[i] = m_device.allocateMemory(ai);
            m_device.bindImageMemory(m_taaImage[i], m_taaMemory[i], 0);

            vk::ImageViewCreateInfo vi{};
            vi.image = m_taaImage[i];
            vi.viewType = vk::ImageViewType::e2D;
            vi.format = vk::Format::eR16G16B16A16Sfloat;
            vi.subresourceRange = vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            m_taaView[i] = m_device.createImageView(vi);
        }

        // Sekali seumur hidup: pindahkan ke eGeneral dan biarkan di sana, sama
        // seperti image storage milik ShadowAAA. Sekaligus dinolkan supaya frame
        // pertama tidak membaca memori sampah sebagai history.
        {
            vk::CommandBufferAllocateInfo cai{};
            cai.commandPool = m_commandPool;
            cai.level = vk::CommandBufferLevel::ePrimary;
            cai.commandBufferCount = 1;
            vk::CommandBuffer cmd = m_device.allocateCommandBuffers(cai)[0];
            vk::CommandBufferBeginInfo bi{};
            bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
            cmd.begin(bi);

            std::array<vk::ImageMemoryBarrier,2> bar{};
            for (int i = 0; i < 2; ++i) {
                bar[i].oldLayout = vk::ImageLayout::eUndefined;
                bar[i].newLayout = vk::ImageLayout::eGeneral;
                bar[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bar[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bar[i].image = m_taaImage[i];
                bar[i].subresourceRange = {vk::ImageAspectFlagBits::eColor,0,1,0,1};
                bar[i].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            }
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr,
                                0, nullptr, 2, bar.data());
            vk::ClearColorValue zero{std::array<float,4>{0.0f,0.0f,0.0f,0.0f}};
            vk::ImageSubresourceRange full{vk::ImageAspectFlagBits::eColor,0,1,0,1};
            for (int i = 0; i < 2; ++i)
                cmd.clearColorImage(m_taaImage[i], vk::ImageLayout::eGeneral, &zero, 1, &full);
            vk::MemoryBarrier post{};
            post.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            post.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader, {},
                                1, &post, 0, nullptr, 0, nullptr);
            cmd.end();
            vk::SubmitInfo si{}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            m_graphicsQueue.submit(si, nullptr);
            m_graphicsQueue.waitIdle();
            m_device.freeCommandBuffers(m_commandPool, 1, &cmd);
        }

        // Pipeline resolve
        const auto CS = vk::ShaderStageFlagBits::eCompute;
        std::array<vk::DescriptorSetLayoutBinding,5> b{};
        for (int i = 0; i < 4; ++i)
            b[i] = {static_cast<uint32_t>(i), vk::DescriptorType::eCombinedImageSampler, 1, CS};
        b[4] = {4, vk::DescriptorType::eStorageImage, 1, CS};
        vk::DescriptorSetLayoutCreateInfo dli{};
        dli.bindingCount = 5; dli.pBindings = b.data();
        m_taaDSL = m_device.createDescriptorSetLayout(dli);

        std::array<vk::DescriptorPoolSize,2> ps{
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 8},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage,         2}};
        vk::DescriptorPoolCreateInfo dpi{};
        dpi.poolSizeCount = 2; dpi.pPoolSizes = ps.data(); dpi.maxSets = 2;
        m_taaPool = m_device.createDescriptorPool(dpi);

        for (int p = 0; p < 2; ++p) {
            vk::DescriptorSetAllocateInfo dsa{};
            dsa.descriptorPool = m_taaPool;
            dsa.descriptorSetCount = 1; dsa.pSetLayouts = &m_taaDSL;
            m_taaSets[p] = m_device.allocateDescriptorSets(dsa)[0];

            vk::Sampler smp = m_shadowAAA.volumetricSampler();
            vk::DescriptorImageInfo iCur {smp, m_hdrImageView,
                                          vk::ImageLayout::eShaderReadOnlyOptimal};
            vk::DescriptorImageInfo iHist{smp, m_taaView[1-p], vk::ImageLayout::eGeneral};
            vk::DescriptorImageInfo iVel {smp, m_shadowAAA.velocityView(),
                                          vk::ImageLayout::eShaderReadOnlyOptimal};
            vk::DescriptorImageInfo iDep {smp, m_shadowAAA.prepassDepthView(),
                                          vk::ImageLayout::eShaderReadOnlyOptimal};
            vk::DescriptorImageInfo iOut {nullptr, m_taaView[p], vk::ImageLayout::eGeneral};

            std::array<vk::WriteDescriptorSet,5> w{};
            w[0] = {m_taaSets[p],0,0,1,vk::DescriptorType::eCombinedImageSampler,&iCur};
            w[1] = {m_taaSets[p],1,0,1,vk::DescriptorType::eCombinedImageSampler,&iHist};
            w[2] = {m_taaSets[p],2,0,1,vk::DescriptorType::eCombinedImageSampler,&iVel};
            w[3] = {m_taaSets[p],3,0,1,vk::DescriptorType::eCombinedImageSampler,&iDep};
            w[4] = {m_taaSets[p],4,0,1,vk::DescriptorType::eStorageImage,&iOut};
            m_device.updateDescriptorSets(5, w.data(), 0, nullptr);
        }

        vk::PushConstantRange pcr{CS, 0, sizeof(glm::vec4)};
        vk::PipelineLayoutCreateInfo pli{};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_taaDSL;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        m_taaPipelineLayout = m_device.createPipelineLayout(pli);

        auto code = readFile("taa_resolve.comp.spv");
        vk::ShaderModule sm = createShaderModule(code);
        vk::PipelineShaderStageCreateInfo st{};
        st.stage = CS; st.module = sm; st.pName = "main";
        vk::ComputePipelineCreateInfo ci{};
        ci.stage = st; ci.layout = m_taaPipelineLayout;
        m_taaPipeline = m_device.createComputePipeline(nullptr, ci).value;
        m_device.destroyShaderModule(sm);
    }

    void createCompositePipeline() {
        auto vc = readFile("composite.vert.spv");
        auto fc = readFile("composite.frag.spv");
        vk::ShaderModule vm = createShaderModule(vc), fm = createShaderModule(fc);

        std::array<vk::PipelineShaderStageCreateInfo,2> stages{};
        stages[0].stage = vk::ShaderStageFlagBits::eVertex;
        stages[0].module = vm; stages[0].pName = "main";
        stages[1].stage = vk::ShaderStageFlagBits::eFragment;
        stages[1].module = fm; stages[1].pName = "main";

        // Tanpa vertex input sama sekali: ketiga verteks segitiga layar penuh
        // dibangkitkan dari gl_VertexIndex di dalam shader.
        vk::PipelineVertexInputStateCreateInfo vi{};
        vk::PipelineInputAssemblyStateCreateInfo ia{};
        ia.topology = vk::PrimitiveTopology::eTriangleList;

        vk::Viewport vp{0.0f, 0.0f, (float)m_swapChainExtent.width,
                        (float)m_swapChainExtent.height, 0.0f, 1.0f};
        vk::Rect2D scr{vk::Offset2D{0,0}, m_swapChainExtent};
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount = 1; vs.pViewports = &vp;
        vs.scissorCount  = 1; vs.pScissors  = &scr;

        vk::PipelineRasterizationStateCreateInfo rz{};
        rz.polygonMode = vk::PolygonMode::eFill;
        rz.lineWidth   = 1.0f;
        rz.cullMode    = vk::CullModeFlagBits::eNone;
        rz.frontFace   = vk::FrontFace::eCounterClockwise;

        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        cba.blendEnable = VK_FALSE;
        vk::PipelineColorBlendStateCreateInfo cb{};
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        // Descriptor: satu sampler untuk target HDR
        vk::DescriptorSetLayoutBinding b0{0, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment};
        vk::DescriptorSetLayoutCreateInfo dli{};
        dli.bindingCount = 1; dli.pBindings = &b0;
        m_compositeDSL = m_device.createDescriptorSetLayout(dli);

        vk::DescriptorPoolSize psz{vk::DescriptorType::eCombinedImageSampler, 2};
        vk::DescriptorPoolCreateInfo dpi{};
        dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz; dpi.maxSets = 2;
        m_compositePool = m_device.createDescriptorPool(dpi);

        // DUA set: komposit membaca history TAA yang baru saja ditulis, dan
        // itu berganti tiap frame mengikuti parity ping-pong.
        for (int p = 0; p < 2; ++p) {
            vk::DescriptorSetAllocateInfo dsa{};
            dsa.descriptorPool = m_compositePool;
            dsa.descriptorSetCount = 1; dsa.pSetLayouts = &m_compositeDSL;
            m_compositeSets[p] = m_device.allocateDescriptorSets(dsa)[0];

            vk::DescriptorImageInfo dii{m_shadowAAA.volumetricSampler(), m_taaView[p],
                                        vk::ImageLayout::eGeneral};
            vk::WriteDescriptorSet w{};
            w.dstSet = m_compositeSets[p]; w.dstBinding = 0; w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            w.pImageInfo = &dii;
            m_device.updateDescriptorSets(1, &w, 0, nullptr);
        }

        // v44: dua vec4 — params lama + parameter FXAA.
        vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eFragment, 0,
                                  static_cast<uint32_t>(2 * sizeof(glm::vec4))};
        vk::PipelineLayoutCreateInfo pli{};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_compositeDSL;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        m_compositePipelineLayout = m_device.createPipelineLayout(pli);

        vk::GraphicsPipelineCreateInfo pi{};
        pi.stageCount = 2; pi.pStages = stages.data();
        pi.pVertexInputState = &vi;  pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vs;  pi.pRasterizationState = &rz;
        pi.pMultisampleState = &ms;  pi.pColorBlendState    = &cb;
        pi.layout     = m_compositePipelineLayout;
        pi.renderPass = m_compositeRenderPass;
        m_compositePipeline = m_device.createGraphicsPipeline(nullptr, pi).value;

        m_device.destroyShaderModule(vm);
        m_device.destroyShaderModule(fm);
    }

    void createFramebuffers() {
        // Main pass: SATU framebuffer saja. Ia menggambar ke target HDR yang
        // tunggal, bukan ke salah satu image swapchain, jadi tidak ada alasan
        // menduplikasinya per image.
        {
            std::array<vk::ImageView,2> att = {m_hdrImageView, m_depthImageView};
            vk::FramebufferCreateInfo fi{};
            fi.renderPass=m_renderPass; fi.attachmentCount=2; fi.pAttachments=att.data();
            fi.width=m_swapChainExtent.width; fi.height=m_swapChainExtent.height; fi.layers=1;
            m_hdrFramebuffer = m_device.createFramebuffer(fi);
        }

        // Pass komposit: satu framebuffer per image swapchain, warna saja.
        m_swapChainFramebuffers.resize(m_swapChainImageViews.size());
        for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
            vk::ImageView att = m_swapChainImageViews[i];
            vk::FramebufferCreateInfo fi{};
            fi.renderPass=m_compositeRenderPass; fi.attachmentCount=1; fi.pAttachments=&att;
            fi.width=m_swapChainExtent.width; fi.height=m_swapChainExtent.height; fi.layers=1;
            m_swapChainFramebuffers[i] = m_device.createFramebuffer(fi);
        }
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo ci{};
        ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        ci.queueFamilyIndex = m_graphicsFamily;
        m_commandPool = m_device.createCommandPool(ci);
    }

    // ==========================================================================
    // BUFFERS
    // ==========================================================================
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags props,
                      vk::Buffer& buf, vk::DeviceMemory& mem) {
        vk::BufferCreateInfo bi{};
        bi.size = size; bi.usage = usage; bi.sharingMode = vk::SharingMode::eExclusive;
        buf = m_device.createBuffer(bi);
        auto mr = m_device.getBufferMemoryRequirements(buf);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, props);
        mem = m_device.allocateMemory(ai);
        m_device.bindBufferMemory(buf, mem, 0);
    }

    void copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo ai{};
        ai.level=vk::CommandBufferLevel::ePrimary; ai.commandPool=m_commandPool; ai.commandBufferCount=1;
        auto cb = m_device.allocateCommandBuffers(ai)[0];
        vk::CommandBufferBeginInfo bi{};
        bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        cb.begin(bi);
        vk::BufferCopy cr{}; cr.size = size;
        cb.copyBuffer(src, dst, 1, &cr);
        cb.end();
        vk::SubmitInfo si{}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        m_graphicsQueue.submit(si, VK_NULL_HANDLE);
        m_graphicsQueue.waitIdle();
        m_device.freeCommandBuffers(m_commandPool, cb);
    }

    void createLevelBuffers() {
        std::vector<Vertex>   verts;
        std::vector<uint32_t> idxs;
        generateLevelGeometry(verts, idxs, m_tiles); // mengisi m_tiles untuk frustum culling
        m_levelIndexCount = (uint32_t)idxs.size();

        // Geometri level tidak pernah bergerak, jadi posisi frame sebelumnya
        // SAMA dengan posisi sekarang. Diisi di satu tempat, tepat sebelum
        // diunggah, supaya menambah generator geometri baru tidak bisa lupa —
        // dan kalau lupa pun gejalanya bukan diam-diam: prevPos nol berarti
        // motion vector raksasa dan permukaannya langsung terlihat kacau.
        for (auto& v : verts) v.prevPos = v.pos;

        // ---- Jarak cascade diturunkan dari geometri, bukan ditebak ----------
        //
        // Kamera hanya bisa berada DI DALAM level, jadi jarak terjauh dari
        // kamera ke geometri mana pun adalah diagonal kotak pembatas scene.
        // Di atas itu ditambahkan panjang bayangan terpanjang yang mungkin:
        // caster tertinggi dibagi tan(elevasi matahari). Elevasinya tetap
        // 12/20 karena matahari mengorbit pada ketinggian tetap — hanya
        // azimutnya yang berputar, jadi bayangan terpanjang tidak berubah.
        //
        // Semua tile ikut dihitung, termasuk mesh OBJ, sehingga menambah aset
        // yang lebih besar akan melebarkan jangkauan dengan sendirinya.
        if (!m_tiles.empty()) {
            glm::vec3 lo( FLT_MAX), hi(-FLT_MAX);
            for (const auto& t : m_tiles) {
                lo = glm::min(lo, t.aabb.min);
                hi = glm::max(hi, t.aabb.max);
            }
            const glm::vec3 size    = hi - lo;
            const float     diag    = glm::length(size);
            const float     sunTan  = 12.0f / 20.0f;          // elevasi orbit matahari
            const float     shadowLen = size.y / std::max(sunTan, 0.05f);

            // Margin 15% menutupi lantai yang menjulur sedikit di luar dinding
            // dan pergerakan kamera di tepi peta.
            m_shadowFitDist = (diag + shadowLen) * 1.15f;
            m_shadowFitDist = std::min(std::max(m_shadowFitDist, 10.0f),
                                       ShadowAAA::Cfg::SHADOW_MAX_DISTANCE);

            LOG_INFO("ShadowAAA",
                "jangkauan cascade dipasang dari geometri: " +
                std::to_string(m_shadowFitDist) + " m (diagonal scene " +
                std::to_string(diag) + " m + bayangan terpanjang " +
                std::to_string(shadowLen) + " m), batas atas " +
                std::to_string(ShadowAAA::Cfg::SHADOW_MAX_DISTANCE) + " m");
        }

        // Inisialisasi m_visibleTiles: semua tile visible di awal
        m_visibleTiles.reserve(m_tiles.size());
        for (auto& t : m_tiles)
            m_visibleTiles.push_back({t.firstIndex, t.indexCount});

        // Bangun per-face DrawCalls untuk FrustumCullingManager (dari main.cpp)
        // Memberikan culling granularitas lebih halus per face (bukan per tile)
        FrustumCullingManager::BuildLevelDrawCalls(verts, idxs, m_levelDrawCalls);
        m_mergedDrawCalls = m_levelDrawCalls; // init: semua visible
        LOG_INFO("FrustumCulling",
            "Built " + std::to_string(m_levelDrawCalls.size()) + " face DrawCalls from " +
            std::to_string(m_tiles.size()) + " tiles");

        vk::DeviceSize vs = sizeof(verts[0]) * verts.size();
        vk::Buffer stgV; vk::DeviceMemory stgVM;
        createBuffer(vs, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stgV, stgVM);
        void* d = m_device.mapMemory(stgVM, 0, vs);
        memcpy(d, verts.data(), vs); m_device.unmapMemory(stgVM);
        createBuffer(vs, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, m_vertexBuffer, m_vertexBufferMemory);
        copyBuffer(stgV, m_vertexBuffer, vs);
        m_device.destroyBuffer(stgV); m_device.freeMemory(stgVM);

        vk::DeviceSize is = sizeof(idxs[0]) * idxs.size();
        vk::Buffer stgI; vk::DeviceMemory stgIM;
        createBuffer(is, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stgI, stgIM);
        d = m_device.mapMemory(stgIM, 0, is);
        memcpy(d, idxs.data(), is); m_device.unmapMemory(stgIM);
        createBuffer(is, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, m_indexBuffer, m_indexBufferMemory);
        copyBuffer(stgI, m_indexBuffer, is);
        m_device.destroyBuffer(stgI); m_device.freeMemory(stgIM);
    }

    // =========================================================================
    //  Bola dinamis: bangun mesh sekali, tulis ulang posisinya tiap frame
    // =========================================================================
    void createBallBuffers() {
        if (!Config::BALL_ENABLED) return;

        const int   SEG = Config::BALL_SEGMENTS;
        const int   RNG = Config::BALL_RINGS;
        const float R   = Config::BALL_RADIUS;
        const glm::vec3 COL(0.72f, 0.30f, 0.26f);   // merah bata, kontras dgn kelabu

        std::vector<uint32_t> idx;
        std::vector<std::vector<uint32_t>> grid;

        // Bola UV. Kutub dibuat sebagai satu baris verteks yang diulang supaya
        // indeksnya seragam; segitiga degenerate di kutub tidak dibuat karena
        // baris kutub ditangani terpisah di bawah.
        for (int r = 0; r <= RNG; ++r) {
            float phi = float(M_PI) * float(r) / float(RNG);
            float y   = std::cos(phi) * R;
            float rad = std::sin(phi) * R;
            std::vector<uint32_t> row;
            for (int sg = 0; sg < SEG; ++sg) {
                float a = 2.0f * float(M_PI) * float(sg) / float(SEG);
                glm::vec3 p(std::cos(a) * rad, y, std::sin(a) * rad);
                row.push_back(static_cast<uint32_t>(m_ballLocalPos.size()));
                m_ballLocalPos.push_back(p);
                Vertex v{};
                v.pos    = p;                       // diisi ulang tiap frame
                v.color  = COL;
                v.normal = glm::normalize(p + glm::vec3(0.0f, 1e-6f, 0.0f));
                m_ballVerts.push_back(v);
            }
            grid.push_back(row);
        }
        for (int r = 0; r < RNG; ++r) {
            for (int sg = 0; sg < SEG; ++sg) {
                int s2 = (sg + 1) % SEG;
                uint32_t a = grid[r][sg],     b = grid[r][s2];
                uint32_t c = grid[r+1][s2],   d = grid[r+1][sg];
                if (r == 0)            { idx.push_back(a); idx.push_back(c); idx.push_back(d); }
                else if (r == RNG - 1) { idx.push_back(a); idx.push_back(b); idx.push_back(c); }
                else { idx.push_back(a); idx.push_back(b); idx.push_back(c);
                       idx.push_back(c); idx.push_back(d); idx.push_back(a); }
            }
        }
        m_ballIndexCount = static_cast<uint32_t>(idx.size());

        // Vertex buffer host-visible, dipetakan permanen.
        vk::DeviceSize vs = sizeof(Vertex) * m_ballVerts.size();
        createBuffer(vs, vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                     m_ballVertexBuffer, m_ballVertexMemory);
        m_ballMapped = m_device.mapMemory(m_ballVertexMemory, 0, vs);

        // Index buffer statis -> device local lewat staging, sama seperti lantai.
        vk::DeviceSize is = sizeof(uint32_t) * idx.size();
        vk::Buffer stg; vk::DeviceMemory stgM;
        createBuffer(is, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent, stg, stgM);
        void* dp = m_device.mapMemory(stgM, 0, is);
        std::memcpy(dp, idx.data(), (size_t)is);
        m_device.unmapMemory(stgM);
        createBuffer(is, vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     m_ballIndexBuffer, m_ballIndexMemory);
        copyBuffer(stg, m_ballIndexBuffer, is);
        m_device.destroyBuffer(stg); m_device.freeMemory(stgM);

        // Dipanggil dua kali dengan sengaja: panggilan pertama mengisi pos,
        // yang kedua menyalin pos itu ke prevPos. Tanpa ini prevPos tetap nol
        // pada frame pertama dan seluruh bola melaporkan motion vector sebesar
        // jarak dari titik asal.
        updateBall(0.0f);
        updateBall(0.0f);

        LOG_INFO("Mesh", "bola dinamis: " + std::to_string(m_ballVerts.size()) +
                 " verteks, " + std::to_string(m_ballIndexCount / 3) + " segitiga, " +
                 "lintasan " + std::to_string(Config::BALL_TRAVEL * 2.0f) +
                 " m tiap " + std::to_string(Config::BALL_PERIOD_SEC) + " detik");
    }

    // Maju-mundur sepanjang sumbu Z. Sinus dipilih, bukan gerak segitiga,
    // supaya kecepatannya kontinu — pembalikan arah yang mendadak akan membuat
    // velocity buffer melompat dan TAA ikut tersentak.
    void updateBall(float time) {
        if (!Config::BALL_ENABLED || !m_ballMapped) return;
        const float w   = 6.28318530718f / Config::BALL_PERIOD_SEC;
        const float off = std::sin(time * w) * Config::BALL_TRAVEL;
        const glm::vec3 c(Config::BALL_X, Config::BALL_Y, Config::BALL_Z + off);

        // prevPos diambil dari pos frame LALU sebelum ditimpa. Inilah yang
        // membuat motion vector bola benar. Pada frame pertama keduanya sama,
        // sehingga velocity nol dan tidak ada lonjakan saat aplikasi mulai.
        for (size_t i = 0; i < m_ballVerts.size(); ++i) {
            m_ballVerts[i].prevPos = m_ballVerts[i].pos;
            m_ballVerts[i].pos     = m_ballLocalPos[i] + c;
        }
        std::memcpy(m_ballMapped, m_ballVerts.data(),
                    sizeof(Vertex) * m_ballVerts.size());
    }

    void createFloorBuffers() {
        // v41: lantai mengikuti ukuran peta, bukan lagi 40x40 m tetap. Peta
        // 90 x 91,5 m di atas lantai 40 m akan menggantung di ruang kosong.
        const float fx = levelMap[0].size() * Config::MAP_SCALE * 0.5f
                       + Config::MAP_FLOOR_MARGIN;
        const float fz = levelMap.size()    * Config::MAP_SCALE * 0.5f
                       + Config::MAP_FLOOR_MARGIN;
        const std::vector<Vertex> fv = {
            {{-fx,-.01f, fz},{.3f,.3f,.3f},{0,1,0}},
            {{ fx,-.01f, fz},{.3f,.3f,.3f},{0,1,0}},
            {{ fx,-.01f,-fz},{.3f,.3f,.3f},{0,1,0}},
            {{-fx,-.01f,-fz},{.3f,.3f,.3f},{0,1,0}}
        };
        const std::vector<uint32_t> fi = {0,1,2,2,3,0};

        // fv memakai inisialisasi agregat yang hanya mengisi tiga anggota
        // pertama, jadi prevPos-nya nol. Lantai diam, jadi harus disamakan
        // dengan pos — kalau tidak, seluruh lantai melaporkan motion vector
        // sebesar 20 meter dan TAA akan menghapusnya jadi bubur.
        std::vector<Vertex> fvp = fv;
        for (auto& v : fvp) v.prevPos = v.pos;

        vk::DeviceSize vs = sizeof(fvp[0])*fvp.size();
        vk::Buffer stgV; vk::DeviceMemory stgVM;
        createBuffer(vs,vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,stgV,stgVM);
        void* d = m_device.mapMemory(stgVM,0,vs); memcpy(d,fvp.data(),vs); m_device.unmapMemory(stgVM);
        createBuffer(vs,vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,m_floorVertexBuffer,m_floorVertexBufferMemory);
        copyBuffer(stgV,m_floorVertexBuffer,vs); m_device.destroyBuffer(stgV); m_device.freeMemory(stgVM);

        vk::DeviceSize is = sizeof(fi[0])*fi.size();
        vk::Buffer stgI; vk::DeviceMemory stgIM;
        createBuffer(is,vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,stgI,stgIM);
        d = m_device.mapMemory(stgIM,0,is); memcpy(d,fi.data(),is); m_device.unmapMemory(stgIM);
        createBuffer(is,vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,m_floorIndexBuffer,m_floorIndexBufferMemory);
        copyBuffer(stgI,m_floorIndexBuffer,is); m_device.destroyBuffer(stgI); m_device.freeMemory(stgIM);
    }

    void createJoystickBuffers() {
        const uint32_t segments = 32;
        std::vector<JoystickVertex> verts;
        verts.push_back({{0.0f, 0.0f}});
        for (uint32_t i = 0; i <= segments; i++) {
            float a = 2.0f * (float)M_PI * i / segments;
            verts.push_back({{cosf(a), sinf(a)}});
        }
        m_joystickVertexCount = (uint32_t)verts.size();
        vk::DeviceSize sz = sizeof(JoystickVertex) * verts.size();
        vk::Buffer stg; vk::DeviceMemory stgM;
        createBuffer(sz,vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,stg,stgM);
        void* d = m_device.mapMemory(stgM,0,sz); memcpy(d,verts.data(),sz); m_device.unmapMemory(stgM);
        createBuffer(sz,vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,m_joystickVertexBuffer,m_joystickVertexBufferMemory);
        copyBuffer(stg,m_joystickVertexBuffer,sz); m_device.destroyBuffer(stg); m_device.freeMemory(stgM);
    }

    void createUniformBuffers() {
        vk::DeviceSize sz = sizeof(UniformBufferObject);
        m_uniformBuffersLevel.resize(m_maxFrames);
        m_uniformBuffersMemoryLevel.resize(m_maxFrames);
        m_uniformBuffersMappedLevel.resize(m_maxFrames);
        m_uniformBuffersFloor.resize(m_maxFrames);
        m_uniformBuffersMemoryFloor.resize(m_maxFrames);
        m_uniformBuffersMappedFloor.resize(m_maxFrames);

        for (size_t i = 0; i < m_maxFrames; i++) {
            createBuffer(sz, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
                m_uniformBuffersLevel[i], m_uniformBuffersMemoryLevel[i]);
            m_uniformBuffersMappedLevel[i] = m_device.mapMemory(m_uniformBuffersMemoryLevel[i],0,sz);
            createBuffer(sz, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
                m_uniformBuffersFloor[i], m_uniformBuffersMemoryFloor[i]);
            m_uniformBuffersMappedFloor[i] = m_device.mapMemory(m_uniformBuffersMemoryFloor[i],0,sz);
        }

        std::array<vk::DescriptorPoolSize,2> ps{};
        ps[0] = {vk::DescriptorType::eUniformBuffer,        (uint32_t)(m_maxFrames*2)};
        // x4, bukan x2: tiap set sekarang punya DUA combined image sampler
        // (shadowFinal di binding 1, light shaft di binding 2), dan ada dua set
        // per frame (level dan floor). Kalau angka ini tidak ikut naik,
        // allocateDescriptorSets gagal dengan ErrorOutOfPoolMemory.
        ps[1] = {vk::DescriptorType::eCombinedImageSampler, (uint32_t)(m_maxFrames*4)};
        vk::DescriptorPoolCreateInfo dpi{};
        dpi.poolSizeCount=2; dpi.pPoolSizes=ps.data(); dpi.maxSets=(uint32_t)(m_maxFrames*2);
        m_descriptorPool = m_device.createDescriptorPool(dpi);

        std::vector<vk::DescriptorSetLayout> layouts(m_maxFrames, m_descriptorSetLayout);
        vk::DescriptorSetAllocateInfo daiL{};
        daiL.descriptorPool=m_descriptorPool; daiL.descriptorSetCount=(uint32_t)m_maxFrames;
        daiL.pSetLayouts=layouts.data();
        m_descriptorSetsLevel = m_device.allocateDescriptorSets(daiL);
        vk::DescriptorSetAllocateInfo daiF = daiL;
        m_descriptorSetsFloor = m_device.allocateDescriptorSets(daiF);

        for (size_t i = 0; i < m_maxFrames; i++) {
            vk::DescriptorBufferInfo biL{m_uniformBuffersLevel[i],0,sizeof(UniformBufferObject)};
            vk::DescriptorBufferInfo biF{m_uniformBuffersFloor[i],0,sizeof(UniformBufferObject)};
            // SHADOW AAA: binding 1 tidak lagi berisi shadow map depth, melainkan
            // shadowFinal (rg16f, screen-space, sudah ter-filter). Layout eGeneral
            // karena image itu storage image yang sengaja dipertahankan di eGeneral
            // agar tidak perlu transisi layout tiap frame.
            vk::DescriptorImageInfo  ii{m_shadowAAA.finalShadowSampler(),
                                        m_shadowAAA.finalShadowView(),
                                        m_shadowAAA.finalShadowLayout()};
            // binding 2 = light shaft. Setengah resolusi; sampler linear yang
            // membesarkannya kembali, jadi tidak perlu pass upsample sendiri.
            vk::DescriptorImageInfo  vv{m_shadowAAA.volumetricSampler(),
                                        m_shadowAAA.volumetricView(),
                                        m_shadowAAA.volumetricLayout()};
            std::array<vk::WriteDescriptorSet,6> writes{};
            writes[0] = {m_descriptorSetsLevel[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                         nullptr, &biL};
            writes[1] = {m_descriptorSetsLevel[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                         &ii};
            writes[2] = {m_descriptorSetsLevel[i], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                         &vv};
            writes[3] = {m_descriptorSetsFloor[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer,
                         nullptr, &biF};
            writes[4] = {m_descriptorSetsFloor[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                         &ii};
            writes[5] = {m_descriptorSetsFloor[i], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                         &vv};
            m_device.updateDescriptorSets(6, writes.data(), 0, nullptr);
        }
    }

    void createCommandBuffers() {
        m_commandBuffers.resize(m_maxFrames);
        vk::CommandBufferAllocateInfo ai{};
        ai.commandPool=m_commandPool; ai.level=vk::CommandBufferLevel::ePrimary;
        ai.commandBufferCount=(uint32_t)m_commandBuffers.size();
        m_commandBuffers = m_device.allocateCommandBuffers(ai);
    }

    // ==========================================================================
    // SYNC OBJECTS — termasuk compute semaphore & fence
    // ==========================================================================
    void createSyncObjects() {
        m_imageAvailableSemaphores.resize(m_maxFrames);
        m_renderFinishedSemaphores.resize(m_maxFrames);
        m_computeFinishedSemaphores.resize(m_maxFrames);
        m_inFlightFences.resize(m_maxFrames);
        m_computeFences.resize(m_maxFrames);

        vk::SemaphoreCreateInfo si{};
        vk::FenceCreateInfo     fi{};
        fi.flags = vk::FenceCreateFlagBits::eSignaled;

        for (size_t i = 0; i < m_maxFrames; i++) {
            m_imageAvailableSemaphores[i]  = m_device.createSemaphore(si);
            m_renderFinishedSemaphores[i]  = m_device.createSemaphore(si);
            m_computeFinishedSemaphores[i] = m_device.createSemaphore(si);
            m_inFlightFences[i]            = m_device.createFence(fi);
            m_computeFences[i]             = m_device.createFence(fi);
        }

        // ── Timeline Semaphore untuk compute sync (Vulkan 1.2) ─────────────
        // FIX: Cek fitur timelineSemaphore sebelum mencoba create, agar validation
        // layer tidak mengeluh "feature not enabled" setiap frame.
        {
            vk::PhysicalDeviceVulkan12Features vk12Features{};
            vk::PhysicalDeviceFeatures2 features2{};
            features2.pNext = &vk12Features;
            m_physicalDevice.getFeatures2(&features2);

            if (vk12Features.timelineSemaphore) {
                vk::SemaphoreTypeCreateInfo timelineTypeCI{};
                timelineTypeCI.semaphoreType = vk::SemaphoreType::eTimeline;
                timelineTypeCI.initialValue  = 0;

                vk::SemaphoreCreateInfo timelineCI{};
                timelineCI.pNext = &timelineTypeCI;

                try {
                    m_computeTimeline.semaphore    = m_device.createSemaphore(timelineCI);
                    m_computeTimeline.currentValue = 0;
                    LOG_INFO("Sync", "Timeline semaphore created (Vulkan 1.2 path)");
                } catch (const std::exception& e) {
                    LOG_WARN("Sync", "Timeline semaphore create failed: " + std::string(e.what())
                        + " — using binary semaphores only");
                }
            } else {
                LOG_INFO("Sync", "timelineSemaphore feature not available — using binary semaphores only");
                // m_computeTimeline.semaphore tetap VK_NULL_HANDLE, drawFrame akan skip signal
            }
        }
    }

    // ==========================================================================
    // CLEANUP
    // ==========================================================================
    void cleanup() {
    
        // ── Destroy IndirectDrawBuffer jika dipakai ──────────────────────────
        if (m_indirectDrawBuffer.commandBuffer) {
            m_device.destroyBuffer(m_indirectDrawBuffer.commandBuffer);
            m_device.freeMemory(m_indirectDrawBuffer.commandMemory);
        }
        if (m_indirectDrawBuffer.countBuffer) {
            m_device.destroyBuffer(m_indirectDrawBuffer.countBuffer);
            m_device.freeMemory(m_indirectDrawBuffer.countMemory);
        }

        // ── Destroy Timeline Semaphore ───────────────────────────────────────
        if (!m_computeTimeline.isNull())
            m_device.destroySemaphore(m_computeTimeline.semaphore);

        if (Config::ENABLE_VALIDATION) {
            auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
                m_instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT");
            if (fn) fn(m_instance,
                       static_cast<VkDebugUtilsMessengerEXT>(m_debugMessenger), nullptr);
        }

        if (m_joystickPipeline)           m_device.destroyPipeline(m_joystickPipeline);
        if (m_joystickPipelineLayout)     m_device.destroyPipelineLayout(m_joystickPipelineLayout);
        if (m_joystickVertexBuffer)       m_device.destroyBuffer(m_joystickVertexBuffer);
        if (m_joystickVertexBufferMemory) m_device.freeMemory(m_joystickVertexBufferMemory);

        // SHADOW AAA: melepas cascade array, buffer screen-space, pipeline
        // compute, descriptor pool, dan UBO-nya sendiri. Menggantikan seluruh
        // pembersihan shadow map lama.
        m_gpuProfiler.destroy();
        m_shadowAAA.destroy();

        m_device.destroyImageView(m_depthImageView);
        m_device.destroyImage(m_depthImage);
        m_device.freeMemory(m_depthImageMemory);

        m_device.destroyDescriptorPool(m_descriptorPool);
        m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);

        for (size_t i = 0; i < m_maxFrames; i++) {
            m_device.unmapMemory(m_uniformBuffersMemoryLevel[i]);
            m_device.destroyBuffer(m_uniformBuffersLevel[i]);
            m_device.freeMemory(m_uniformBuffersMemoryLevel[i]);
            m_device.unmapMemory(m_uniformBuffersMemoryFloor[i]);
            m_device.destroyBuffer(m_uniformBuffersFloor[i]);
            m_device.freeMemory(m_uniformBuffersMemoryFloor[i]);

            m_device.destroySemaphore(m_renderFinishedSemaphores[i]);
            m_device.destroySemaphore(m_imageAvailableSemaphores[i]);
            m_device.destroySemaphore(m_computeFinishedSemaphores[i]);
            m_device.destroyFence(m_inFlightFences[i]);
            m_device.destroyFence(m_computeFences[i]);
        }

        m_device.destroyBuffer(m_floorIndexBuffer);    m_device.freeMemory(m_floorIndexBufferMemory);
        if (m_ballMapped) { m_device.unmapMemory(m_ballVertexMemory); m_ballMapped = nullptr; }
        if (m_ballVertexBuffer) { m_device.destroyBuffer(m_ballVertexBuffer); m_device.freeMemory(m_ballVertexMemory); }
        if (m_ballIndexBuffer)  { m_device.destroyBuffer(m_ballIndexBuffer);  m_device.freeMemory(m_ballIndexMemory); }
        m_device.destroyBuffer(m_floorVertexBuffer);   m_device.freeMemory(m_floorVertexBufferMemory);
        m_device.destroyBuffer(m_indexBuffer);         m_device.freeMemory(m_indexBufferMemory);
        m_device.destroyBuffer(m_vertexBuffer);        m_device.freeMemory(m_vertexBufferMemory);

        m_device.destroyCommandPool(m_commandPool);
        m_device.destroyCommandPool(m_computeCommandPool);

        // --- TAA ---
        if (m_taaPipeline)       m_device.destroyPipeline(m_taaPipeline);
        if (m_taaPipelineLayout) m_device.destroyPipelineLayout(m_taaPipelineLayout);
        if (m_taaPool)           m_device.destroyDescriptorPool(m_taaPool);
        if (m_taaDSL)            m_device.destroyDescriptorSetLayout(m_taaDSL);
        for (int i = 0; i < 2; ++i) {
            if (m_taaView[i])   m_device.destroyImageView(m_taaView[i]);
            if (m_taaImage[i])  m_device.destroyImage(m_taaImage[i]);
            if (m_taaMemory[i]) m_device.freeMemory(m_taaMemory[i]);
        }

        // --- Fase 3: target HDR + pass komposit ---
        if (m_compositePipeline)       m_device.destroyPipeline(m_compositePipeline);
        if (m_compositePipelineLayout) m_device.destroyPipelineLayout(m_compositePipelineLayout);
        if (m_compositePool)           m_device.destroyDescriptorPool(m_compositePool);
        if (m_compositeDSL)            m_device.destroyDescriptorSetLayout(m_compositeDSL);
        if (m_compositeRenderPass)     m_device.destroyRenderPass(m_compositeRenderPass);
        if (m_hdrFramebuffer)          m_device.destroyFramebuffer(m_hdrFramebuffer);
        if (m_hdrImageView)            m_device.destroyImageView(m_hdrImageView);
        if (m_hdrImage)                m_device.destroyImage(m_hdrImage);
        if (m_hdrMemory)               m_device.freeMemory(m_hdrMemory);

        for (auto fb : m_swapChainFramebuffers) m_device.destroyFramebuffer(fb);
        if (m_skyPipeline) m_device.destroyPipeline(m_skyPipeline);
        m_device.destroyPipeline(m_graphicsPipeline);
        m_device.destroyPipelineLayout(m_pipelineLayout);
        m_device.destroyRenderPass(m_renderPass);
        for (auto iv : m_swapChainImageViews) m_device.destroyImageView(iv);
        m_device.destroySwapchainKHR(m_swapChain);
        m_device.destroy();
        m_instance.destroySurfaceKHR(m_surface);
        m_instance.destroy();
        SDL_DestroyWindow(window);
        SDL_Quit();

        LOG_INFO("App", "Cleanup complete — all systems released");
    }
};

// =============================================================================
// ENTRY POINT
// =============================================================================
// Tanda tangan HARUS (int, char**). SDL_main.h mendefinisikan ulang `main`
// menjadi SDL_main, dan SDL memanggilnya dengan argc/argv — `int main()` tanpa
// argumen akan gagal tautan di Android.
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    try {
        LOG_INFO("Main", "Starting " + Config::APP_NAME);
        HelloTriangleApplication app;
        app.run();
        LOG_INFO("Main", "Application terminated normally");
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        // v74: SDL_LogCritical, bukan std::cerr — di Android std::cerr dibuang,
        // dan pesan inilah satu-satunya petunjuk kenapa aplikasi mati.
        SDL_LogCritical(SDL_LOG_CATEGORY_APPLICATION, "[FATAL] %s", e.what());
        return EXIT_FAILURE;
    }
}
