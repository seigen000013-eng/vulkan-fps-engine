// =============================================================================
// shaders.cpp — Vulkan FPS Engine, Shadow Pipeline AAA (Fase 1 PRD)
//
// File ini BUKAN untuk dikompilasi compiler C++. Ini kumpulan sumber GLSL;
// pisahkan tiap blok jadi file sendiri dengan nama sesuai judulnya, lalu
// kompilasi ke SPIR-V.
//
// Nama file .spv HARUS persis seperti ini (dicari runtime oleh main2.cpp):
//
//   glslc --target-env=vulkan1.3 -O prepass.vert         -o prepass.vert.spv
//   glslc --target-env=vulkan1.3 -O prepass.frag         -o prepass.frag.spv
//   glslc --target-env=vulkan1.3 -O csm_shadow.vert      -o csm_shadow.vert.spv
//   glslc --target-env=vulkan1.3 -O csm_shadow.frag      -o csm_shadow.frag.spv
//   glslc --target-env=vulkan1.3 -O ao_resolve.comp      -o ao_resolve.comp.spv
//   glslc --target-env=vulkan1.3 -O volumetric.comp      -o volumetric.comp.spv
//   glslc --target-env=vulkan1.3 -O csm_resolve.comp     -o csm_resolve.comp.spv
//   glslc --target-env=vulkan1.3 -O shadow_temporal.comp -o shadow_temporal.comp.spv
//   glslc --target-env=vulkan1.3 -O svgf_atrous.comp      -o svgf_atrous.comp.spv
//   glslc --target-env=vulkan1.3 -O main.vert            -o main.vert.spv
//   glslc --target-env=vulkan1.3 -O main.frag            -o main.frag.spv
//   glslc --target-env=vulkan1.3 -O sky.vert             -o sky.vert.spv
//   glslc --target-env=vulkan1.3 -O sky.frag             -o sky.frag.spv
//   glslc --target-env=vulkan1.3 -O taa_resolve.comp     -o taa_resolve.comp.spv
//   glslc --target-env=vulkan1.3 -O composite.vert       -o composite.vert.spv
//   glslc --target-env=vulkan1.3 -O composite.frag       -o composite.frag.spv
//   glslc --target-env=vulkan1.3 -O joystick.vert        -o joystick.vert.spv
//   glslc --target-env=vulkan1.3 -O joystick.frag        -o joystick.frag.spv
//
// shadow.vert / shadow.frag lama SUDAH TIDAK DIPAKAI — digantikan csm_shadow.*
//
// Alur data:
//   prepass      -> depth + gNormal(oct rg16f) + velocity(rg16f)
//   csm_shadow   -> shadowCascadeArray D32, 4 layer
//   csm_resolve  -> shadowResolve  rg16f   (.r=shadow .g=penumbraWorld)
//   temporal     -> shadowHistory  rgba16f (.b=guideWorld .a=dev) ping-pong
//   blur_h       -> shadowBlurTmp  rgba16f
//   blur_v       -> shadowFinal    rg16f   <- dibaca main.frag di binding 1
// =============================================================================

common_ubo.glsl:
// =============================================================================
//  SUMBER TUNGGAL untuk ShadowUBO dan pembantu yang bergantung padanya.
//
//  Sebelum berkas ini ada, blok di bawah disalin utuh ke LIMA shader:
//  ao_resolve, csm_resolve, shadow_temporal, svgf_atrous, volumetric.
//  22 field x 5 salinan = 110 baris yang harus selalu sinkron.
//
//  Yang membuatnya berbahaya bukan panjangnya, melainkan cara ia gagal:
//  menambah satu field lalu lupa di salah satu shader BUKAN compile error.
//  Layout std140 bergeser, dan gejalanya muncul sebagai nilai acak di shader
//  yang salah — kelas bug yang paling mahal dikejar.
//
//  Sisipkan dengan  #include "common_ubo.glsl"  sesudah baris #version.
//  Tidak punya prasyarat.
// =============================================================================
layout(std140, binding = 0) uniform ShadowUBO {
    mat4 cascadeVP[6];
    mat4 invViewProj;
    mat4 prevViewProj;
    mat4 view;
    mat4 viewProj;              // ray march contact shadow
    vec4 cascadeSplitView[2];   // 6 cascade tidak muat di satu vec4 -> [c>>2][c&3]
    vec4 cascadeExtentWorld[2];
    vec4 cascadeDepthRange[2];
    vec4 cascadeTexelWorld[2];
    vec4 lightDirWorld;
    vec4 cameraPos;      // w = tan(fovY/2)
    vec4 resolution;     // xy = res, zw = 1/res
    vec4 params0;        // x=sunTan y=minPenumbra z=maxPenumbra w=normalOffsetScale
    vec4 params1;        // x=depthBiasScale y=temporalBlend z=disoccRel w=blurMaxRadiusPx
    vec4 params2;        // x=frameIndex y=nearZ z=farZ w=cascadeBlendFrac
    vec4 params3;        // x=shadowRes y=1/shadowRes z=blockerSearchWorld w=blendStatic
    vec4 params4;        // x=deltaTime y=minBlurRadiusPx z=minPcfScreenPx w=pakaiBilinearManual
    vec4 params5;        // x=debugMode y=jarakBayanganEfektif z=fadeFrac
    vec4 params6;        // contact: x=panjang y=langkah z=tebal w=jarakFade
    vec4 params7;   // AO: x=radius y=slices z=steps w=intensitas
    vec4 params8;   // x=aoMinVisibility y=aoBlurRadiusPx z=aoMaxRadiusPx
    vec4 params9;   // x=normalOffsetMaxWorld y=depthBiasMaxWorld
} u;

float linearizeDepth(float d) {
    float n = u.params2.y, f = u.params2.z;
    return (n * f) / max(f - d * (f - n), 1e-6);
}

// 6 cascade tidak muat di satu vec4, jadi diindeks [c>>2][c&3].
float splitOf  (int c) { return u.cascadeSplitView  [c >> 2][c & 3]; }
float extentOf (int c) { return u.cascadeExtentWorld[c >> 2][c & 3]; }
float rangeOf  (int c) { return u.cascadeDepthRange [c >> 2][c & 3]; }
float texelOf  (int c) { return u.cascadeTexelWorld [c >> 2][c & 3]; }

common_oct.glsl:
// =============================================================================
//  Pengemasan normal oktahedral. Murni aritmetika, tanpa prasyarat apa pun.
//  Dulu octEncode ada di prepass.frag dan ao_resolve; octDecode di ao_resolve
//  dan csm_resolve; dan svgf_atrous punya salinan ketiga bernama octDec.
//  Nama octDec sekarang dihapus — pemakainya memakai octDecode.
// =============================================================================
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                        n.y >= 0.0 ? 1.0 : -1.0);
    }
    return n.xy;
}

vec3 octDecode(vec2 f) {
    vec3  n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

common_noise.glsl:
// =============================================================================
//  Blue noise + putaran deret emas per frame.
//
//  PRASYARAT: sampler2D bernama blueNoiseTex HARUS sudah dideklarasikan sebelum
//  baris #include ini. Nomor binding-nya berbeda di tiap shader, jadi ia tidak
//  bisa ikut pindah ke sini — letakkan #include sesudah deklarasi sampler.
// =============================================================================
const float GOLDEN_FRAC = 0.61803399;

float blueNoise(ivec2 px, float frameIdx) {
    float bn = texelFetch(blueNoiseTex, px & ivec2(127), 0).r;
    return fract(bn + frameIdx * GOLDEN_FRAC);
}

prepass.vert:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// prepass.vert  —  Depth Pre-Pass + G-Normal + Velocity
// Push constant 128 byte (batas aman semua vendor): viewProj + prevViewProj.
// Model matrix diasumsikan identity (level statis), sesuai main2.cpp.
// ============================================================================

layout(push_constant) uniform PC {
    mat4 viewProj;      // frame ini,        TANPA jitter
    mat4 prevViewProj;  // frame sebelumnya, TANPA jitter
} pc;

// Hanya location 0 dan 2 yang dideklarasikan. Pipeline pre-pass di C++ juga
// hanya mengirim dua atribut ini — kalau dikirim location 1 (warna) yang tidak
// dipakai, validation layer melapor "attribute not consumed by vertex shader".
layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inPrevPos;   // posisi frame lalu, ruang dunia

layout(location = 0) out vec3 vNormalWorld;
layout(location = 1) out vec4 vClipCurr;
layout(location = 2) out vec4 vClipPrev;

void main() {
    vec4 worldPos = vec4(inPosition, 1.0);

    vClipCurr = pc.viewProj     * worldPos;
    // Motion vector dihitung dari posisi frame LALU yang sebenarnya, bukan dari
    // posisi sekarang yang diproyeksikan dengan matriks kamera lama. Untuk
    // geometri statis keduanya identik karena prevPos diisi sama dengan pos.
    // Untuk benda bergerak keduanya berbeda, dan selisih itulah gerakan objek
    // itu sendiri — tanpa ini TAA menyangka permukaannya diam lalu memadukannya
    // dengan riwayat dari tempat yang salah.
    vClipPrev = pc.prevViewProj * vec4(inPrevPos, 1.0);

    vNormalWorld = inNormal;
    gl_Position  = vClipCurr;
}

prepass.frag:

#version 450
#include "common_oct.glsl"

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// prepass.frag  —  MRT: [0] oct-encoded normal (rg16f), [1] velocity (rg16f)
// Depth ditulis otomatis oleh rasterizer ke attachment depth.
// ============================================================================

layout(location = 0) in vec3 vNormalWorld;
layout(location = 1) in vec4 vClipCurr;
layout(location = 2) in vec4 vClipPrev;

layout(location = 0) out vec2 outNormalOct;
layout(location = 1) out vec2 outVelocity;

// Octahedral normal encoding (Cigolle et al.) — 2 kanal, error < 0.1 derajat

void main() {
    vec3 n = normalize(vNormalWorld);
    outNormalOct = octEncode(n);

    // NDC -> UV. Konvensi Vulkan: proj[1][1] sudah dinegasikan di C++,
    // sehingga ndc.y = -1 berada di baris atas, sama dengan gl_FragCoord.y = 0.
    // Jadi uv = ndc * 0.5 + 0.5 konsisten dengan gl_FragCoord.xy / resolution.
    vec2 uvCurr = (vClipCurr.xy / vClipCurr.w) * 0.5 + 0.5;
    vec2 uvPrev = (vClipPrev.xy / vClipPrev.w) * 0.5 + 0.5;

    // Motion vector di ruang UV: dari pixel sekarang ke posisi yang sama frame lalu.
    // Reprojection nanti memakai prevUV = uv - velocity.
    outVelocity = uvCurr - uvPrev;
}

csm_shadow.vert:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// csm_shadow.vert  —  Render satu cascade ke satu layer texture array.
// Satu render pass per cascade; matriks cascade dikirim lewat push constant.
// ============================================================================

layout(push_constant) uniform PC {
    mat4 cascadeViewProj;
    vec4 lightAndScale;   // xyz = arah MENUJU cahaya, w = ukuran texel cascade (m)
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec3 inNormal;

// ============================================================================
// NORMAL OFFSET DI SISI CASTER, bukan sisi penerima.
//
// Ini perbedaan ARSITEKTURAL, bukan penyetelan — dan inilah yang menyisakan
// sobekan selama dua puluh iterasi.
//
// Sebelumnya offset diterapkan saat MEMBACA: titik sampel penerima digeser
// sepanjang normalnya, sebesar worldPerPixel. Masalahnya worldPerPixel berubah
// menurut kedalaman TIAP PIKSEL. Di dinding yang menjauh, kedalaman berubah
// menyeberangi layar, jadi besar pergeseran ikut berubah — dan tepi bayangan
// tidak pernah bisa jadi garis lurus, sekecil apa pun offsetnya dibuat.
// Itu sebabnya menurunkannya dari 3 texel ke 1 texel mengurangi sobek tapi
// tidak menghapusnya.
//
// Sekarang offset diterapkan sekali saat MENULIS shadow map: verteks caster
// digeser sepanjang normalnya sebelum diproyeksikan. Pergeserannya per-verteks
// dan seragam, jadi siluet di dalam shadow map tetap berupa garis lurus —
// cuma sedikit bergeser. Sisi penerima kini menyampel di posisi persis, tanpa
// pergeseran sama sekali.
// ============================================================================
void main() {
    vec3 L = normalize(pc.lightAndScale.xyz);
    vec3 N = normalize(inNormal);

    float NdotL = dot(N, L);
    float sinT  = sqrt(max(1.0 - NdotL * NdotL, 0.0));
    float slope = clamp(sinT / max(abs(NdotL), 0.05), 0.0, 3.0);

    // Dibatasi satu texel, sama seperti kode pembanding. Di atas itu, distorsi
    // siluetnya mulai terlihat sebagai bayangan yang lepas dari kaki benda.
    float texel  = pc.lightAndScale.w;
    float offset = min(texel * slope * 0.5, texel);

    // DIGESER SEPANJANG L, BUKAN SEPANJANG N.
    //
    // Menggeser sepanjang normal MEROBEK geometri di setiap rusuk. Sebabnya ada
    // di generator level: addFace() membuat 4 verteks BARU tiap kali dipanggil,
    // jadi dua muka yang bertemu di satu rusuk tidak berbagi verteks sama
    // sekali — rusuk itu diwakili dua pasang verteks terpisah. Ketika masing
    // masing digeser sepanjang normalnya sendiri, keduanya bergerak ke arah
    // berbeda dan rusuknya terbuka. Pada rusuk cembung 90 derajat celahnya
    // selebar offset*sqrt(2), yaitu 0,39 cm di cascade 0 sampai 10,2 cm di
    // cascade 3 — dan konsisten sekitar 1-1,6 piksel layar di semua cascade.
    //
    // Lewat celah itu matahari menembus ke permukaan yang seharusnya berbayang
    // penuh. Terukur di tangkapan layar: gumpalan ~7x10 px di rusuk pertemuan
    // dua dinding, dan cahaya yang ditambahkan berwarna (0,992 1,000 1,023)
    // alias PUTIH NETRAL — bukan ambient bernada langit, bukan pula light shaft
    // yang hangat. Celah mentahnya cuma 1-2 px; a-trous menyebarkannya jadi
    // gumpalan, dan edge-stopping tidak menahannya karena kebocorannya terjadi
    // di atas satu permukaan yang menerus.
    //
    // Menggeser sepanjang L menghapus masalahnya tanpa kehilangan apa pun:
    //   - Proyeksi cascade ortografik memandang persis sepanjang -L, jadi
    //     translasi sepanjang L TIDAK mengubah koordinat XY di shadow map sama
    //     sekali. Siluetnya jadi siluet sejati, tanpa distorsi — ini bahkan
    //     lebih baik daripada tujuan awal memindahkan offset ke sisi caster.
    //   - Yang berubah hanya kedalaman yang tersimpan, dan besarnya dibuat
    //     PERSIS SAMA dengan sebelumnya: menggeser sepanjang N sejauh offset
    //     mengubah kedalaman sebesar offset*NdotL, jadi di sini digeser
    //     sepanjang L sejauh offset*NdotL. Bias-nya identik.
    //   - Karena XY tidak bergeser, muka-muka yang bersebelahan tetap rapat.
    //     Tidak ada lagi celah untuk ditembus cahaya; yang tersisa di rusuk
    //     cuma beda nilai kedalaman kecil, dan itu tidak berlubang.
    vec3 p = inPosition + L * (offset * NdotL);
    gl_Position = pc.cascadeViewProj * vec4(p, 1.0);
}

csm_shadow.frag:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// csm_shadow.frag  —  Depth-only. Rasterizer yang menulis gl_FragDepth.
// Dibiarkan kosong; dipertahankan agar pipeline punya fragment stage eksplisit
// (beberapa driver mobile/Linux rewel dengan pipeline tanpa FS + depth bias).
// ============================================================================

void main() {
}

ao_resolve.comp:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// ao_resolve.comp — Ground-Truth Ambient Occlusion (GTAO)
//
//   INI SHADOW, BUKAN LIGHTING. Ambient occlusion adalah bayangan dari cahaya
//   langit: berapa banyak kubah langit yang terhalang geometri di sekitar satu
//   titik. Bedanya dengan CSM cuma sumber cahayanya — CSM membayangi matahari
//   (satu arah), AO membayangi langit (semua arah).
//
//   Kenapa ini celah terbesar yang tersisa: di engine ini ambient masih konstan
//   rata (ambientStrength x 1.0) untuk SETIAP piksel. Artinya bagian dalam
//   bayangan benar-benar datar tanpa struktur, dan sudut pertemuan dinding
//   dengan lantai sama terangnya dengan tengah lantai. Itu yang membuat
//   bayangan terbaca "menempel di atas gambar" alih-alih membumi. Tidak ada
//   penyetelan PCSS yang bisa memperbaikinya, karena masalahnya bukan di
//   bayangan matahari.
//
//   Metode: GTAO (Jimenez et al. 2016) — mencari sudut horizon di beberapa
//   irisan bidang, lalu mengintegralkan visibilitas kosinus secara analitik.
//   Berbeda dari SSAO klasik yang menghitung rasio sampel tertutup, integral
//   GTAO adalah solusi tertutup sehingga hasilnya benar secara fisik dan jauh
//   lebih sedikit noise untuk jumlah sampel yang sama.
//
//   Keluaran r16f: 1.0 = terbuka penuh, 0.0 = tertutup penuh.
//   Peredamannya menumpang rantai yang sudah ada — temporal reprojection lalu
//   bilateral blur, sama persis seperti bayangan CSM.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common_ubo.glsl"
#include "common_oct.glsl"

layout(binding = 1) uniform sampler2D sceneDepth;
layout(binding = 2) uniform sampler2D gNormalOct;
layout(binding = 3, rgba16f) uniform writeonly image2D aoOut;   // .r AO, .gb bent normal (oct)
layout(binding = 4) uniform sampler2D blueNoiseTex;   // LUT 128x128 R8
#include "common_noise.glsl"

const float PI          = 3.14159265;
// 64 -> 128. Batas KEDUA, dan selama ia 64 menaikkan AO_MAX_RADIUS_PX di C++
// tidak berpengaruh apa pun: radiusPx = min(diminta, min(params8.z, MAX_RAD_PX)),
// jadi yang terkecil dari keduanya yang berlaku. Sapuan menunjukkan oklusi
// jenuh di 128 px, jadi kedua batas disamakan ke sana.
const float MAX_RAD_PX  = 128.0;  // batas atas radius layar, menjaga cache

// v68: 4 -> 8 dan 8 -> 12. Ini BATAS ATAS perulangan, bukan jumlah yang
// dipakai — Cfg::AO_SLICES dan Cfg::AO_STEPS yang menentukan. Selama batas ini
// 4/8, menaikkan angka di main2.cpp tidak berpengaruh apa pun: `break` di dalam
// loop memotongnya diam-diam, tanpa error maupun peringatan.
const int   MAX_SLICES  = 8;
const int   MAX_STEPS   = 12;

// v68: ketebalan penghalang yang masih dianggap benar-benar menutupi. Di atas
// ini sampel dianggap benda lain yang lewat di depan.
const float THICKNESS_WORLD = 0.35;


// ---- BLUE NOISE ------------------------------------------------------------
// Menggantikan interleaved gradient noise.
//
// IGN murah dan bagus secara temporal, tapi spektrum SPASIALnya mengandung
// energi frekuensi rendah — sampel yang berdekatan cenderung mirip, sehingga
// noise-nya menggumpal jadi bercak. Gumpalan frekuensi rendah persis yang
// PALING SULIT dihapus filter mana pun: a-trous meredam frekuensi tinggi.
//
// Blue noise memindahkan seluruh energinya ke frekuensi tinggi. Sisa noise-nya
// jadi butiran halus yang mudah dihapus, dan kalaupun tersisa, mata jauh lebih
// sulit melihatnya daripada bercak.
//
// Tile 128x128 bersifat toroidal, sampler REPEAT jadi tidak menghasilkan seam.
// Pergeseran per frame memakai rasio emas: urutan yang paling merata untuk
// sembarang jumlah frame, jadi akumulasi temporal tidak pernah mengulang pola.


// Rekonstruksi posisi view-space.
// CATATAN sumbu Y: proj[1][1] dinegasikan di C++, sehingga
//     ndc.y = -viewY / (tanV * linZ)   ->   viewY = -ndc.y * tanV * linZ
// Tanda minus itu WAJIB. Tanpanya seluruh perhitungan horizon tercermin
// vertikal dan AO jadi salah di permukaan mendatar seperti lantai.
vec3 viewPosFrom(ivec2 c, float linZ) {
    vec2  uv   = (vec2(c) + 0.5) * u.resolution.zw;
    float tanV = u.cameraPos.w;
    float tanH = tanV * (u.resolution.x * u.resolution.w);
    vec2  ndc  = uv * 2.0 - 1.0;
    return vec3(ndc.x * tanH * linZ, -ndc.y * tanV * linZ, -linZ);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim   = ivec2(u.resolution.xy);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    int slices = int(u.params7.y);
    int steps  = int(u.params7.z);

    float rawD = texelFetch(sceneDepth, coord, 0).r;
    if (rawD >= 1.0 || slices <= 0) {          // langit, atau AO dimatikan
        // Tanpa oklusi, arah langit rata-rata = normal permukaan itu sendiri.
        vec3 nw = octDecode(texelFetch(gNormalOct, coord, 0).rg);
        imageStore(aoOut, coord, vec4(1.0, octEncode(nw), 0.0));
        return;
    }

    float viewZ = linearizeDepth(rawD);
    vec3  P     = viewPosFrom(coord, viewZ);
    vec3  N     = normalize(mat3(u.view) * octDecode(texelFetch(gNormalOct, coord, 0).rg));
    vec3  V     = normalize(-P);

    float radiusWorld = u.params7.x;

    // Radius dunia -> radius layar. Dekat kamera radiusnya besar dalam piksel,
    // jauh mengecil — itu benar: AO adalah efek berskala DUNIA, bukan layar.
    float worldPerPixel = 2.0 * viewZ * u.cameraPos.w * u.resolution.w;
    float radiusPx = min(radiusWorld / max(worldPerPixel, 1e-6),
                         min(u.params8.z, MAX_RAD_PX));
    if (radiusPx < 2.0) {                       // terlalu jauh untuk berarti
        // N di sini sudah di ruang view; bent normal disimpan di ruang DUNIA,
        // jadi harus dikembalikan dulu.
        imageStore(aoOut, coord,
                   vec4(1.0, octEncode(normalize(transpose(mat3(u.view)) * N)), 0.0));
        return;
    }

    // Noise dirotasi per frame — sama seperti Vogel disk di csm_resolve, supaya
    // akumulasi temporal punya sampel baru tiap frame alih-alih mengulang pola.
    float frameOffset = fract(u.params2.x * 0.61803399);
    float noiseDir    = blueNoise(coord, u.params2.x);
    // Texel yang BERJAUHAN, bukan pergeseran fase.
    //
    // Kalau keduanya membaca texel yang sama, noiseStep selalu = noiseDir + c.
    // Arah irisan dan jarak langkah lalu bergerak bersamaan, dan hasilnya pola
    // terstruktur — bukan sebaran acak. Itu tampak sebagai noise berpola.
    float noiseStep   = blueNoise(coord + ivec2(37, 89), u.params2.x);

    float visibility = 0.0;

    // BENT NORMAL — arah rata-rata langit yang MASIH TERBUKA.
    //
    // AO menjawab "berapa banyak langit yang terhalang". Bent normal menjawab
    // "dari arah mana sisa langitnya datang". Di ruang terbuka ia sama dengan
    // normal permukaan; di dekat dinding ia condong menjauhi dinding itu.
    //
    // Ini yang membuat ambient punya ARAH, bukan cuma intensitas. Tanpanya,
    // sudut ruangan cuma jadi lebih gelap; dengannya, sudut ruangan juga
    // berubah warna karena cahaya yang tersisa datang dari langit di atas,
    // bukan dari dinding di sampingnya.
    vec3 bentSum = vec3(0.0);

    for (int s = 0; s < MAX_SLICES; ++s) {
        if (s >= slices) break;

        float phi = (float(s) + noiseDir) * PI / float(slices);
        vec2  dir = vec2(cos(phi), sin(phi));          // arah LANGKAH di ruang layar

        // ====================================================================
        // SUMBU Y LAYAR BERLAWANAN DENGAN SUMBU Y VIEW.
        //
        // dir dipakai untuk melangkah di koordinat piksel, tempat +y menunjuk
        // KE BAWAH. Basis bidang irisan hidup di ruang view, tempat +y menunjuk
        // KE ATAS. Memakai vec3(dir, 0) apa adanya membuat vektor tangen
        // menunjuk berlawanan dengan arah langkah untuk setiap irisan yang punya
        // komponen y — sehingga sisi +dir dan -dir TERTUKAR saat sudut horizon
        // diberi tanda.
        //
        // Akibatnya bukan sedikit meleset, melainkan runtuh total: pada lantai
        // datar tanpa penghalang sekalipun, sampel yang sebidang dibaca sebagai
        // penghalang. Uji numerik memberi visibilitas 1,543 kalau tandanya benar
        // dan 0,008 kalau tertukar — beda 190 kali. Itu yang membuat seluruh
        // bagian dalam bayangan menjadi hitam pekat.
        // ====================================================================
        vec3 dirView = vec3(dir.x, -dir.y, 0.0);

        vec3 axis = cross(dirView, V);
        float axisLen = length(axis);
        if (axisLen < 1e-5) continue;
        axis /= axisLen;

        vec3  projN    = N - axis * dot(N, axis);
        float projNLen = length(projN);
        if (projNLen < 1e-4) continue;
        vec3 projNn = projN / projNLen;

        // Sudut bertanda normal terproyeksi terhadap arah pandang
        vec3  tangent = cross(V, axis);
        float n = atan(dot(projNn, tangent), dot(projNn, V));

        // ---- Pencarian horizon dua arah ----------------------------------
        float cosH1 = -1.0;   // arah -dir
        float cosH2 = -1.0;   // arah +dir

        for (int side = 0; side < 2; ++side) {
            float sgn  = (side == 0) ? -1.0 : 1.0;
            float best = -1.0;

            for (int st = 0; st < MAX_STEPS; ++st) {
                if (st >= steps) break;

                // Jarak LINIER, bukan kuadratik.
                //
                // Distribusi kuadratik memang merapatkan sampel di dekat titik
                // asal, tapi konsekuensinya celah antar sampel di ujung membesar
                // drastis: dengan 8 langkah dan radius 64 px, celah terakhir
                // belasan piksel. Horizon yang jatuh di celah itu tidak pernah
                // tersampel, jadi sudut horizon meloncat dan tepi AO patah
                // bertangga — itulah "sobekan" di kaki dinding.
                // Linier memberi celah seragam; noiseStep menggeser seluruh
                // rangkaian tiap piksel dan tiap frame sehingga sisa celahnya
                // menjadi noise yang bisa diredam, bukan tangga yang menetap.
                // v68: JARAK ANTAR-SAMPEL EKSPONENSIAL, bukan linear.
                //
                // Pembagian linear menyebar sampel merata sepanjang radius.
                // Pada radius 128 px dengan 8 langkah, celahnya 16 piksel — dan
                // horizon yang jatuh di antara dua sampel TIDAK PERNAH terlihat.
                // Itu sumber tepi AO yang meloncat bertangga.
                //
                // Oklusi yang paling menentukan justru datang dari yang DEKAT:
                // sudut pertemuan dinding-lantai, celah sempit, kaki benda.
                // Pangkat 1,7 merapatkan sampel di dekat piksel dan
                // merenggangkannya di ujung, tempat kontribusinya toh sudah
                // diredam oleh atten.
                //
                // Terukur pada radius 128 px:
                //   linear 8 langkah : celah merata 16,0 px
                //   eksponensial 12  : sampel pertama 0,6 px
                float tLin = (float(st) + noiseStep) / float(steps);
                float t    = pow(tLin, 1.7);

                vec2  sc2 = vec2(coord) + dir * (sgn * t * radiusPx);
                ivec2 sc  = clamp(ivec2(sc2), ivec2(0), dim - 1);
                if (sc == coord) continue;

                float sd = texelFetch(sceneDepth, sc, 0).r;
                if (sd >= 1.0) continue;               // langit tidak menutupi

                vec3  S     = viewPosFrom(sc, linearizeDepth(sd));
                vec3  delta = S - P;
                float len   = length(delta);
                if (len < 1e-4) continue;

                float cosA = dot(delta, V) / len;

                // Peluruhan jarak: penghalang di luar radius diabaikan mulus,
                // supaya tidak ada tepi keras saat objek melewati batas radius.
                float atten = clamp(1.0 - (len * len) / (radiusWorld * radiusWorld),
                                    0.0, 1.0);
                // v68: UJI KETEBALAN.
                //
                // GTAO polos menganggap setiap penghalang tak berhingga tebal:
                // sekali sebuah sampel menutupi horizon, ia dianggap menutup
                // selamanya. Akibatnya tiang tipis, pagar, dan tepi atap
                // menghasilkan bayangan gelap jauh lebih besar daripada
                // bendanya sendiri — gumpalan gelap yang menempel di siluet.
                //
                // Sampel yang berada jauh LEBIH DEKAT ke kamera daripada piksel
                // ini kemungkinan besar benda lain yang cuma lewat di depan,
                // bukan dinding yang benar-benar menutupi. Kontribusinya diredam
                // sebanding seberapa jauh ia di depan.
                float inFront = max(0.0, viewZ - (-S.z));
                float thin    = 1.0 - smoothstep(THICKNESS_WORLD,
                                                 THICKNESS_WORLD * 3.0, inFront);
                cosA = mix(-1.0, cosA, atten * thin);

                best = max(best, cosA);
            }

            if (side == 0) cosH1 = best; else cosH2 = best;
        }

        float h1 = -acos(clamp(cosH1, -1.0, 1.0));
        float h2 =  acos(clamp(cosH2, -1.0, 1.0));

        // Jepit horizon ke hemisfer normal
        h1 = n + max(h1 - n, -PI * 0.5);
        h2 = n + min(h2 - n,  PI * 0.5);

        // ---- Integral visibilitas kosinus GTAO (bentuk tertutup) ---------
        float sinN = sin(n);
        float cosN = cos(n);
        float inner = 0.25 * (-cos(2.0 * h1 - n) + cosN + 2.0 * h1 * sinN)
                    + 0.25 * (-cos(2.0 * h2 - n) + cosN + 2.0 * h2 * sinN);

        visibility += projNLen * inner;

        // Arah tengah busur yang tidak terhalang, di dalam bidang irisan ini.
        // Bobotnya sama dengan sumbangan visibilitasnya, sehingga irisan yang
        // lebih terbuka lebih menentukan arah akhirnya.
        float hMid = 0.5 * (h1 + h2);
        vec3  arcDir = normalize(V * cos(hMid) + tangent * sin(hMid));
        bentSum += arcDir * max(projNLen * inner, 0.0);
    }

    float ao = clamp(visibility / float(slices), 0.0, 1.0);
    ao = pow(ao, max(u.params7.w, 0.01));      // intensitas

    // Lantai visibilitas. AO yang boleh mencapai nol akan menghitamkan bagian
    // dalam bayangan sepenuhnya, karena di sana ambient adalah SATU-SATUNYA
    // sumber cahaya. Bayangan kelas AAA tidak pernah pekat: pada referensi yang
    // diukur, bayangan menahan sekitar 31% kecerahan area terang dan masih
    // menyisakan detail di dalamnya. Lantai ini menjamin hal yang sama secara
    // struktural, bukan bergantung pada penyetelan yang pas.
    ao = mix(u.params8.x, 1.0, ao);

    // Kembali ke ruang dunia, dan dijaga tetap di hemisfer normal: bent normal
    // yang menembus permukaan tidak punya arti fisik.
    vec3 bentView = (dot(bentSum, bentSum) > 1e-8) ? normalize(bentSum) : N;
    if (dot(bentView, N) < 0.0) bentView = N;
    vec3 bentWorld = normalize(transpose(mat3(u.view)) * bentView);

    imageStore(aoOut, coord, vec4(ao, octEncode(bentWorld), 0.0));
}

csm_resolve.comp:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// csm_resolve.comp  — v2
//
//   PERUBAHAN DARI v1:
//   [1] Batas bawah filter TIDAK lagi dinyatakan dalam texel shadow map.
//       Itu penyebab pita "mengikuti cascade": 1,5 texel = 3,5 mm di cascade 0
//       tapi 47 mm di cascade 3, jadi kelembutan melompat 13x di setiap batas
//       cascade. Sekarang batas bawah dijaga di ruang LAYAR oleh pass blur,
//       yang tidak tahu apa-apa soal cascade sehingga mustahil bikin pita.
//   [2] Fallback cascade tidak lagi lompat keras — ia lewat jalur blending yang
//       sama, dan indeks cascade tidak dirusak (v1 menimpa `ci` lalu memakai
//       cascadeSplitView[ci-1] yang sudah salah).
//   [3] Sample turun 16->8 (blocker) dan 24->12 (PCF), plus early-out sub-texel.
//       Akumulasi temporal yang menutupi kekurangannya.
//
//   REVISI v3 — perbaikan atas [1] yang kebablasan:
//       v2 menurunkan batas bawah PCF ke 1 texel dan menyerahkan seluruh
//       pelembutan minimum ke blur layar. Itu salah kaprah: blur layar
//       MELEMBUTKAN tepi tapi tidak MEMINDAHKAN posisinya, jadi kuantisasi texel
//       shadow map lolos utuh — terukur sebagai tangga 1 piksel setiap ~14 piksel
//       di tepi diagonal (itu "gerigi"-nya). Yang menghapus tangga hanya PCF yang
//       merata-rata beberapa texel.
//       Sekarang batas bawah PCF hidup lagi, TAPI dinyatakan dalam piksel layar:
//           minFilterWorld = MIN_PCF_SCREEN_PX * worldPerPixel(viewZ)
//       worldPerPixel adalah fungsi kontinu dari kedalaman, bukan fungsi cascade,
//       jadi nilainya tidak melompat di batas cascade. Pita cascade tetap tidak
//       muncul, tapi tangga texel hilang.
//
//   REVISI v4 — bilinear manual untuk driver tanpa hardware PCF:
//       Terukur di build v3: tepi bayangan datar sempurna (+-0,15 px) sepanjang
//       80 piksel, lalu naik 1,6 px dalam 5 piksel. Itu tanda posisi tepi
//       TERKUNCI ke kelipatan texel, bukan sekadar kurang lembut. Melebarkan
//       filter tidak akan menolong: yang salah posisinya, bukan kelembutannya.
//       Penyebabnya sampler compare yang jatuh ke NEAREST karena driver tidak
//       melaporkan SAMPLED_IMAGE_FILTER_LINEAR untuk format depth — umum di GPU
//       mobile. Tanpa interpolasi, tiap tap compare menghasilkan 0 atau 1 mentah
//       terhadap satu texel, sehingga tepi tidak punya resolusi sub-texel.
//       cmpTap() di bawah melakukan interpolasi 2x2 itu sendiri saat hardware
//       tidak menyediakannya (params4.w = 1). Kalau hardware mendukung, jalur
//       murahnya tetap dipakai — percabangannya seragam untuk seluruh dispatch,
//       jadi tidak ada divergensi.
//
//   REVISI v5 — dither sub-texel (hipotesis v4 TERBANTAH):
//       Log dari perangkat melaporkan hardware 2x2 PCF TERSEDIA, jadi penyebab
//       yang saya duga di v4 keliru. Pengukuran ulang: tepi datar di y=425,50
//       (simpangan baku 0,08-0,14 px) lalu meloncat 1,11 px ke y=426,61 — dua
//       dataran yang sangat bersih. Satu texel di jarak itu = 1,27 px, jadi
//       loncatannya tepat SATU TEXEL.
//
//       Pemahaman yang benar: bilinear PCF menginterpolasi HASIL PERBANDINGAN
//       di dalam satu texel, jadi nilainya mulus. Tapi siluet caster di shadow
//       map dirasterisasi tanpa cakupan parsial — depth buffer tidak mengenal
//       "setengah texel tertutup". Jadi POSISI rampnya tetap terkunci ke kisi
//       texel. Bilinear menghaluskan nilai, bukan posisi. Melebarkan filter cuma
//       menolong kalau ia mencakup BANYAK texel; di 2,7 texel jelas belum cukup.
//
//       Obatnya: geser seluruh disk PCF sejauh pecahan texel secara acak per
//       piksel per frame. Undakan yang tadinya koheren berubah jadi noise
//       berfrekuensi tinggi, lalu akumulasi temporal dan blur bilateral
//       meratakannya jadi ramp yang benar-benar mulus. Ini pendekatan stokastik
//       yang sama yang dipakai denoiser bayangan modern.
//
//   REVISI v6 — BIAS dibuat kontinu melintasi cascade:
//       Loncatan 1,57 px masih tersisa, dengan kedua dataran sangat bersih
//       (simpangan baku 0,068 dan 0,052 px). Dither sub-texel tidak bisa
//       menyentuhnya, jadi sumbernya bukan kuantisasi texel.
//
//       Biangnya: normal-offset bias dan depth bias masih diturunkan dari
//       cascadeTexelWorld, yang MELOMPAT di batas cascade (11,7 mm di cascade 2
//       vs 31 mm di cascade 3). Normal offset menggeser titik sample menjauhi
//       permukaan; kalau besarnya meloncat, POSISI tepi bayangan ikut meloncat.
//       Perkiraan kasar untuk dinding di ~10 m: selisih offset ~42 mm, dan satu
//       piksel di jarak itu ~11,5 mm — jadi loncatan beberapa piksel. Ordenya
//       persis cocok dengan yang terukur.
//
//       Sekarang KEDUA bias diturunkan dari worldPerPixel, sama seperti batas
//       bawah filter di v3. Setelah ini tidak ada satu pun besaran di jalur
//       resolve yang bergantung pada cascadeTexelWorld — jadi secara struktural
//       tidak ada lagi yang bisa meloncat di batas cascade.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common_ubo.glsl"
#include "common_oct.glsl"

layout(binding = 1) uniform sampler2D            sceneDepth;
layout(binding = 2) uniform sampler2D            gNormalOct;
layout(binding = 3) uniform sampler2DArray       shadowCascades;
layout(binding = 4) uniform sampler2DArrayShadow shadowCascadesCmp;
layout(binding = 5, rg16f) uniform writeonly image2D shadowResolveOut;
layout(binding = 6) uniform sampler2D blueNoiseTex;   // LUT 128x128 R8
#include "common_noise.glsl"

const int   NUM_CASCADES    = 6;
const int   BLOCKER_SAMPLES = 8;
const int   PCF_SAMPLES     = 12;   // jalur hardware PCF
const int   PCF_SAMPLES_BILINEAR = 6;   // jalur bilinear manual (4 fetch/tap)
// 28 -> 64, dan minimumnya dinaikkan lewat TAP_SPACING_TEXELS di bawah.
//
// Ini mengikuti kode pembanding, dan alasannya struktural bukan selera:
// pass blur terpisah untuk bayangan shadow map BUKAN hal standar. Penyaringan
// bayangan seharusnya terjadi di dalam kernel PCF. Blur terpisah lazimnya untuk
// AO dan untuk bayangan ray-traced, yang keluarannya stokastik.
//
// Dengan 12 tap plus dither, keluaran PCF berisik, dan kekurangan itu ditambal
// blur dan akumulasi temporal. Dua tambalan itulah yang menggeser posisi tepi —
// blur lewat radius yang berubah menurut posisi, temporal lewat variance
// clipping yang tidak simetris. Menaikkan jumlah tap menghapus kebutuhan
// keduanya, bukan menambal akibatnya.
const int   PCF_SAMPLES_MAX = 64;

// Panjang maksimum sumbu elips, dalam texel. Nilainya TIDAK bebas: jarak antar
// tap = 2 * ANISO_MAX_TEXELS / jumlahTap, dan satu tap compare 2x2 cuma menutup
// sekitar satu texel. Kalau jaraknya melebihi itu, ada LUBANG di antara tap —
// dan tepi bayangan yang menyeberangi lubang akan melompat. Itu persis gerigi.
//   2 * 10 / 28 = 0,71 texel  -> aman, tidak ada lubang.
const float ANISO_MAX_TEXELS   = 10.0;
// 0.9 -> 0.55: tap dirapatkan, jadi jumlah tap yang dipakai naik untuk lebar
// kernel yang sama. Inilah yang membuat hasil PCF cukup mulus tanpa denoiser.
const float TAP_SPACING_TEXELS = 0.55;
// Amplitudo dither dalam texel. 0.75 cukup memecah undakan 1 texel tanpa
// menambah kekaburan yang terasa. Naikkan kalau undakan masih terlihat.
// TEXEL_JITTER 0,75 -> 0,10.
//
// Pergeseran sub-texel acak ini dipasang untuk memecah pita kisi texel. Diukur,
// ia TIDAK melakukan itu — ia menaikkan derau DAN pita sekaligus, di setiap
// radius filter yang diuji. Simulasi PCF (tepi occluder + cakram Vogel + putaran
// blue noise per piksel, 400 percobaan per titik):
//
//   R=2,5 texel, 10 tap      derau     pita
//     jitter 0,00           0,0192    0,0010
//     jitter 0,30           0,0292    0,0017
//     jitter 0,75           0,0586    0,0034     <- nilai lama
//
//   R=4 texel, 16 tap : 0,0159 (j=0) -> 0,0428 (j=0,75)   2,7x lebih berisik
//   R=8 texel, 32 tap : 0,0121 (j=0) -> 0,0259 (j=0,75)   2,1x lebih berisik
//
// Sebabnya jelas begitu dilihat: putaran cakram Vogel SUDAH acak per piksel
// lewat blue noise, dan itu saja sudah memutus keterkaitan antar piksel
// bertetangga. Menambah translasi acak di atasnya tidak menambah dekorelasi apa
// pun — ia cuma menambah varians.
//
// Lebih dari itu, tiap tap di sini memakai perbandingan bilinear perangkat keras
// ("Hardware 2x2 PCF: TERSEDIA" di log), jadi keluaran tiap tap sudah berupa ramp
// mulus selebar satu texel. Tidak ada kuantisasi yang perlu di-dither. Jitter
// baru masuk akal pada perbandingan BINER tanpa bilinear, dan bahkan di sana
// simulasi menunjukkan ia tetap merugikan (0,0329 -> 0,0657).
//
// Tidak dinolkan sepenuhnya: 0,10 menyisakan sedikit pemecah pola untuk kasus
// yang tidak tercakup model ini (mis. bila jalur bilinear perangkat keras tak
// tersedia di perangkat lain), dengan biaya derau yang hampir nol.
const float TEXEL_JITTER    = 0.10;
const float GOLDEN_ANGLE    = 2.39996323;

// Skalar per-cascade tersimpan sebagai vec4[2]; helper ini menyembunyikan
// pengindeksannya supaya sisa shader tetap terbaca.


// ---- BLUE NOISE ------------------------------------------------------------
// Menggantikan interleaved gradient noise.
//
// IGN murah dan bagus secara temporal, tapi spektrum SPASIALnya mengandung
// energi frekuensi rendah — sampel yang berdekatan cenderung mirip, sehingga
// noise-nya menggumpal jadi bercak. Gumpalan frekuensi rendah persis yang
// PALING SULIT dihapus filter mana pun: a-trous meredam frekuensi tinggi.
//
// Blue noise memindahkan seluruh energinya ke frekuensi tinggi. Sisa noise-nya
// jadi butiran halus yang mudah dihapus, dan kalaupun tersisa, mata jauh lebih
// sulit melihatnya daripada bercak.
//
// Tile 128x128 bersifat toroidal, sampler REPEAT jadi tidak menghasilkan seam.
// Pergeseran per frame memakai rasio emas: urutan yang paling merata untuk
// sembarang jumlah frame, jadi akumulasi temporal tidak pernah mengulang pola.

// Satu tap PCF dengan resolusi sub-texel dijamin.
//   params4.w = 0 -> hardware yang menginterpolasi (1 fetch)
//   params4.w = 1 -> kita interpolasi sendiri  (4 fetch)
float cmpTap(vec2 uv, float layer, float ref) {
    if (u.params4.w < 0.5) {
        return texture(shadowCascadesCmp, vec4(uv, layer, ref));
    }
    float invRes = u.params3.y;
    vec2  t    = uv * u.params3.x - 0.5;
    vec2  f    = fract(t);
    vec2  base = (floor(t) + 0.5) * invRes;

    float s00 = texture(shadowCascadesCmp, vec4(base,                        layer, ref));
    float s10 = texture(shadowCascadesCmp, vec4(base + vec2(invRes, 0.0),    layer, ref));
    float s01 = texture(shadowCascadesCmp, vec4(base + vec2(0.0, invRes),    layer, ref));
    float s11 = texture(shadowCascadesCmp, vec4(base + vec2(invRes, invRes), layer, ref));

    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

// Akar kuadrat matriks 2x2 semi-definit positif, bentuk tertutup.
// Dipakai mengubah kovarians kernel menjadi transformasi yang memetakan
// cakram satuan ke elips yang benar. Tanpa ini kita cuma bisa membuat
// lingkaran, dan lingkaran itu bentuk yang SALAH untuk jejak satu piksel.
mat2 sqrtPSD2(mat2 C) {
    float det = max(C[0][0] * C[1][1] - C[0][1] * C[1][0], 0.0);
    float s   = sqrt(det);
    float t   = sqrt(max(C[0][0] + C[1][1] + 2.0 * s, 1e-20));
    mat2  R   = C;
    R[0][0] += s;
    R[1][1] += s;
    return R / t;
}

vec2 vogelDisk(int i, int count, float rotation) {
    float r     = sqrt(float(i) + 0.5) / sqrt(float(count));
    float theta = GOLDEN_ANGLE * float(i) + rotation;
    return r * vec2(cos(theta), sin(theta));
}

// ---------------------------------------------------------------------------
// CONTACT SHADOW — ray march pendek di ruang layar, menuju cahaya.
//
//   Shadow map seberapa pun resolusinya punya batas: satu texel cascade 0 di
//   sini ~0,4 cm, dan detail kontak yang lebih halus dari itu (celah tipis,
//   kaki benda kecil, pertemuan dua permukaan) hilang. Ray march di buffer
//   depth yang sudah ada mengisi persis celah itu.
//
//   Sengaja tidak jadi pass terpisah: depth, normal, dan posisi dunia semuanya
//   sudah ada di sini, jadi nol image dan nol dispatch tambahan. Hasilnya juga
//   otomatis ikut diredam temporal dan blur bersama bayangan CSM — itu penting,
//   karena ray march dengan langkah sedikit selalu berisik.
//
//   Batasannya jujur: ia hanya tahu apa yang terlihat kamera. Penghalang di
//   luar layar tidak menghasilkan contact shadow. Itu sebabnya ia PELENGKAP
//   CSM, bukan pengganti.
// ---------------------------------------------------------------------------
float contactShadow(vec3 worldPos, vec3 N, vec3 L, float viewZ, float jitter) {
    int steps = int(u.params6.y);
    if (steps <= 0) return 0.0;                 // CONTACT_STEPS = 0 -> fitur mati

    float NdotL = dot(N, L);

    // Permukaan yang menyerempet cahaya adalah kasus TERBURUK untuk ray march:
    // raynya berjalan nyaris sejajar permukaan sehingga terus-menerus menyampel
    // permukaannya sendiri. Kebetulan di situ pula contact shadow tidak berguna,
    // karena CSM sudah menggelapkannya lewat suku NdotL. Jadi dilewati saja.
    if (NdotL < 0.15) return 0.0;

    // Meredup di kejauhan: satu langkah ray di sana lebih kecil daripada satu
    // piksel, jadi yang tersisa cuma noise.
    float fade = 1.0 - smoothstep(u.params6.w * 0.6, u.params6.w, viewZ);
    if (fade <= 0.0) return 0.0;

    float maxLen  = u.params6.x;
    float thick   = u.params6.z;
    float stepLen = maxLen / float(steps);

    float worldPerPixel = 2.0 * viewZ * u.cameraPos.w * u.resolution.w;

    // Offset awal sepanjang normal, diskalakan dengan jejak piksel. Nilai tetap
    // dalam meter tidak cukup: di 20 m, satu piksel saja sudah ~2 cm.
    float offset = max(stepLen * 0.5, worldPerPixel * 2.0);

    // Seberapa menyerong permukaan ini dilihat kamera. Dipakai memperkirakan
    // beda kedalaman yang WAJAR untuk sampel yang masih di bidang yang sama.
    vec3  V     = normalize(u.cameraPos.xyz - worldPos);
    float NdotV = max(abs(dot(N, V)), 0.15);

    vec3 p = worldPos + N * offset + L * (stepLen * jitter);

    for (int i = 0; i < steps; ++i) {
        p += L * stepLen;

        vec4 clip = u.viewProj * vec4(p, 1.0);
        if (clip.w <= 0.0) break;
        vec2 uv2 = (clip.xy / clip.w) * 0.5 + 0.5;
        if (any(lessThan(uv2, vec2(0.0))) || any(greaterThan(uv2, vec2(1.0)))) break;

        // texelFetch, BUKAN texture(). sceneDepth di-bind dengan sampler LINEAR,
        // jadi texture() menginterpolasi kedalaman antar piksel — dan di siluet,
        // hasil interpolasi itu tidak sesuai permukaan mana pun yang nyata.
        // Ray march jadi "menabrak" kedalaman hantu di setiap tepi geometri.
        // Bug ini masih ada di v10 dan ikut menyumbang garis-garis kemarin.
        ivec2 sc = clamp(ivec2(uv2 * u.resolution.xy), ivec2(0), ivec2(u.resolution.xy) - 1);
        float sceneD = texelFetch(sceneDepth, sc, 0).r;
        if (sceneD >= 1.0) continue;                       // langit

        float sceneZ = linearizeDepth(sceneD);

        // ---- Penjagaan SUDUT dan siluet ---------------------------------
        // Uji depth cuma menjawab "ada geometri di depan titik ray, dilihat
        // dari KAMERA". Yang sebenarnya ingin diketahui: "ada geometri antara
        // titik ini dan MATAHARI". Keduanya berbeda, dan paling sering meleset
        // tepat di sudut bangunan: ray berjalan 18 cm, melewati sudut, lalu
        // jatuh ke piksel milik dinding tegak lurus di sebelahnya. Kalau
        // dinding itu kebetulan lebih dekat ke kamera, ia dibaca sebagai
        // penghalang — padahal ia tidak menghalangi matahari sama sekali.
        //
        // Ray hanya bisa berpindah sejauh maxLen dari titik asalnya, jadi
        // kedalamannya pun tidak mungkin berubah lebih dari itu. Piksel yang
        // jauh lebih dekat ke kamera daripada batas tersebut pasti milik
        // permukaan LAIN yang kebetulan lewat di depan — bukan penghalang
        // lokal. Itu yang dibuang di sini.
        if (sceneZ < viewZ - maxLen * 1.5) continue;

        // Penjagaan kedua, memakai normal yang sudah tersedia di pass ini.
        // Penghalang sejati untuk detail sekecil ini praktis selalu permukaan
        // yang searah atau memunggungi cahaya. Permukaan yang justru menghadap
        // matahari LEBIH tegak daripada titik asal hampir pasti dinding
        // bersebelahan di sudut, bukan penghalang.
        vec3 nHit = octDecode(texelFetch(gNormalOct, sc, 0).rg);
        if (dot(nHit, L) > NdotL + 0.25) continue;
        float rayZ   = -(u.view * vec4(p, 1.0)).z;
        float diff   = rayZ - sceneZ;

        // ---- Penolakan tabrakan-diri ------------------------------------
        // INI yang hilang di v9 dan menghasilkan garis-garis diagonal itu.
        //
        // Titik ray sekarang berada setinggi h di atas bidang permukaan:
        //     h = offset + jarak_tempuh * NdotL
        // Kalau piksel yang disampel ternyata BIDANG YANG SAMA, beda
        // kedalamannya bukan nol melainkan sekitar h / NdotV — dan nilai itu
        // TUMBUH tiap langkah. Ambang tetap 1 cm yang dipakai v9 langsung
        // terlampaui setelah beberapa langkah, jadi ray "menabrak" lantainya
        // sendiri secara berkala. Periodisitas itulah yang terlihat sebagai
        // garis sejajar tiap ~7 piksel.
        // PERBAIKAN: jarak tempuh harus memasukkan jitter awal. Titik awal ray
        // sudah digeser L*stepLen*jitter sebelum perulangan, jadi jarak
        // sesungguhnya (i+1+jitter)*stepLen. Tanpa suku jitter, h dinilai
        // terlalu kecil sehingga ambang penolakan tabrakan-diri jadi terlalu
        // longgar — persis celah yang meloloskan garis-garis di v9/v10.
        float travelled = stepLen * (float(i + 1) + jitter);
        float h         = offset + travelled * NdotL;
        float selfDiff  = h / NdotV;

        float minDiff = max(selfDiff * 1.5, viewZ * 0.004);
        if (diff > minDiff && diff < thick + selfDiff) {
            // Peredupan menurut jarak tempuh, bukan nilai biner.
            //
            // Ray berhenti setelah maxLen. Kalau hasilnya biner, ada batas
            // keras di tempat ray kehabisan panjang: penghalang tepat di ujung
            // menggelapkan penuh, penghalang sejengkal di belakangnya tidak
            // menggelapkan sama sekali. Batas itu terlihat sebagai garis tegas
            // yang tidak berhubungan dengan geometri apa pun.
            float tail = 1.0 - smoothstep(0.6, 1.0, travelled / max(maxLen, 1e-4));
            return fade * tail;
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Mengembalikan vec3(shadow, penumbraWorld, valid)
//   valid = 0.0 berarti titik ini jatuh di luar frustum cascade tersebut.
// ---------------------------------------------------------------------------
vec3 sampleCascadePCSS(int ci, vec3 worldPos, vec3 N, float NdotL, float rotation,
                       float worldPerPixel, vec2 texJitter) {
    float extentWorld = extentOf(ci);
    float depthRange  = rangeOf(ci);

    // Normal offset dalam METER, diturunkan dari ukuran piksel layar.
    // SENGAJA tidak memakai cascadeTexelWorld: nilai itu meloncat di batas
    // cascade, dan karena offset ini menggeser titik sample, lompatannya
    // langsung menggeser posisi tepi bayangan.
    // ---- TIDAK ADA pergeseran posisi di sisi penerima ----------------------
    //
    // Normal offset sekarang dikerjakan sekali saat MENULIS shadow map, di
    // csm_shadow.vert. Menggesernya lagi di sini berarti menggeser dua kali —
    // dan pergeseran sisi penerima itu justru yang tidak pernah bisa
    // menghasilkan tepi lurus, karena besarnya berubah menurut kedalaman tiap
    // piksel.
    //
    // Sampel diambil di posisi PERSIS. Yang tersisa cuma depth bias dan RPDB,
    // dan keduanya hanya mengubah NILAI yang dibandingkan.
    vec4  lightClip   = u.cascadeVP[ci] * vec4(worldPos, 1.0);

    vec3  proj  = lightClip.xyz / lightClip.w;
    vec2  uv    = proj.xy * 0.5 + 0.5;
    float zRecv = proj.z;

    if (zRecv <= 0.0 || zRecv >= 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return vec3(0.0, 0.0, 0.0);          // invalid
    }

    // Depth bias juga dijaga tetap dalam METER, baru dikonversi ke satuan depth
    // cascade. Dengan begitu besarnya di dunia nyata sama di semua cascade.
    float slope     = clamp(1.0 - NdotL, 0.0, 1.0);
    // Suku kemiringan turun dari 3.0 ke 0.5.
    //
    // Dulu ia besar karena harus menutup selisih kedalaman menyeberangi seluruh
    // kernel pada permukaan miring. RPDB di bawah sekarang menutup selisih itu
    // per tap, secara tepat. Membiarkan suku lama tetap besar berarti membayar
    // dua kali untuk masalah yang sama — dan bias berlebih itulah yang menggeser
    // posisi tepi bayangan.
    float biasWorld = min(u.params1.x * worldPerPixel * (1.0 + 0.5 * slope),
                          u.params9.y);
    float bias      = clamp(biasWorld / max(depthRange, 1e-4), 1e-6, 0.01);

    // --- Pass 1: blocker search, DUA TAHAP ---------------------------------
    //
    // Tahap pertama memakai radius tetap BLOCKER_SEARCH_WORLD (0,5 m). Radius
    // tetap itu terlalu besar di dekat titik kontak: cakramnya menyapu bagian
    // caster yang jauh lebih tinggi daripada yang benar-benar menaungi penerima,
    // sehingga avgBlocker terseret menjauh dan penumbra jadi terlalu lebar.
    //
    // Diukur terhadap kebenaran analitik pada tiang 0,5 x 4,0 m, matahari 31
    // derajat, distWorld sepanjang bayangan:
    //
    //   x dari kaki   benar    satu tahap        dua tahap
    //     0,13 m      0,155    0,317 (2,04x)     0,157 (1,01x)
    //     0,33 m      0,389    0,548 (1,41x)     0,390 (1,00x)
    //     0,67 m      0,777    0,875 (1,13x)     0,779 (1,00x)
    //     6,00 m      6,997    7,016 (1,00x)     7,001 (1,00x)
    //
    //   galat rata-rata 24,1% -> 0,11%;  di titik kontak 104% -> 1%
    //
    // Tahap kedua mengulang pencarian dengan radius = lebar penumbra taksiran
    // tahap pertama. Itu persis radius yang secara fisik bisa menaungi titik
    // ini, jadi cakramnya berhenti menyapu caster yang tidak relevan. Batas
    // bawahnya dua texel supaya tetap ada yang tersampel.
    //
    // Putarannya digeser setengah sudut emas supaya sampel tahap dua tidak
    // berkorelasi dengan tahap satu — terukur menurunkan galat 0,23% -> 0,11%.
    //
    // Biayanya BLOCKER_SAMPLES tambahan (8 tap). Bandingkan dengan piramida
    // min-kedalaman yang memberi manfaat serupa: 34 MB memori, 72 image view,
    // dan satu pass compute baru. Ini menjawab cacat yang sama dengan 8 fetch.
    float searchUV = u.params3.z / max(extentWorld, 1e-4);
    searchUV = max(searchUV, 2.0 * u.params3.y);

    float avgBlocker = 0.0;
    float distWorld  = 0.0;
    bool  anyBlocker = false;

    for (int pass = 0; pass < 2; ++pass) {
        float sumBlocker = 0.0;
        float weightSum  = 0.0;
        float rot        = rotation + (pass == 1 ? GOLDEN_ANGLE * 0.5 : 0.0);

        for (int i = 0; i < BLOCKER_SAMPLES; ++i) {
            vec2 o = vogelDisk(i, BLOCKER_SAMPLES, rot);
            float d = texture(shadowCascades, vec3(uv + o * searchUV, float(ci))).r;
            if (d < zRecv - bias) {
                float w = dot(o, o) + 0.25;
                sumBlocker += d * w;
                weightSum  += w;
            }
        }

        // Tidak ada blocker di tahap pertama -> benar-benar tanpa bayangan.
        // Di tahap kedua, radius yang menyempit bisa saja meleset dari blocker
        // yang tipis; kalau begitu taksiran tahap pertama yang dipakai, bukan
        // dibuang jadi "tanpa bayangan".
        if (weightSum <= 0.0) {
            if (pass == 0) return vec3(0.0, 0.0, 1.0);
            break;
        }

        anyBlocker = true;
        avgBlocker = sumBlocker / weightSum;
        distWorld  = max((zRecv - avgBlocker) * depthRange, 0.0);

        if (pass == 0) {
            // Radius tahap dua = lebar penumbra yang tersirat dari tahap satu.
            float refineWorld = clamp(distWorld * u.params0.x,
                                      u.params0.y, u.params3.z);
            searchUV = max(refineWorld / max(extentWorld, 1e-4),
                           2.0 * u.params3.y);
        }
    }

    // --- Lebar penumbra untuk directional light ----------------------------
    float penumbraWorld = distWorld * u.params0.x;
    penumbraWorld = clamp(penumbraWorld, u.params0.y, u.params0.z);

    // --- Pass 2: PCF --------------------------------------------------------
    // Batas bawah dalam METER, diturunkan dari ukuran piksel layar di kedalaman
    // ini. Kontinu terhadap kedalaman -> tidak bisa bikin pita cascade; lebarnya
    // ~3 texel -> tangga texel terhapus.
    float filterWorld = max(penumbraWorld, u.params4.z * worldPerPixel);
    float filterUV    = max(filterWorld / max(extentWorld, 1e-4), u.params3.y);

    // Kalau radius filter tidak sampai satu texel, 12 sample Vogel semuanya
    // jatuh di texel yang sama. Satu tap compare sudah cukup dan menghemat
    // 11 fetch — signifikan di GPU mobile.
    // Dither sub-texel HANYA untuk tahap PCF. Blocker search sengaja memakai uv
    // asli supaya estimasi penumbra tetap stabil dan tidak ikut berisik.
    vec2 uvPcf = uv + texJitter;

    if (filterUV <= 1.2 * u.params3.y) {
        return vec3(1.0 - cmpTap(uvPcf, float(ci), zRecv - bias), penumbraWorld, 1.0);
    }

    // ====================================================================
    // KERNEL ANISOTROPIK — bentuk kernel mengikuti JEJAK PIKSEL, bukan lingkaran
    //
    // Satu piksel layar yang jatuh di lantai menyerong menutupi area dunia yang
    // panjang dan tipis, dan di shadow map area itu jadi ELIPS memanjang.
    // Kernel lingkaran punya dua pilihan buruk di situ: cukup besar untuk
    // menutupi sumbu panjang (dan mengaburkan sumbu pendek), atau pas di sumbu
    // pendek (dan kekurangan sampel di sumbu panjang). Yang kedua itu gerigi:
    // tepi bayangan melompat saat menyeberangi celah antar tap.
    //
    // Jejaknya dihitung analitik, bukan dengan ddx/ddy — ini compute shader,
    // tidak ada turunan layar. Perturbasi arah pandang sebesar satu piksel
    // diproyeksikan ke BIDANG permukaan, lalu dibawa ke ruang shadow map.
    // ====================================================================
    vec3  camRight = vec3(u.view[0][0], u.view[1][0], u.view[2][0]);
    vec3  camUp    = vec3(u.view[0][1], u.view[1][1], u.view[2][1]);
    float tanV     = u.cameraPos.w;
    float tanH     = tanV * (u.resolution.x * u.resolution.w);

    vec3  toCam = u.cameraPos.xyz - worldPos;
    float dist  = max(length(toCam), 1e-4);
    vec3  rd    = -toCam / dist;                    // arah kamera -> permukaan

    // Dijaga: pada permukaan yang nyaris sejajar pandangan, pembaginya menuju
    // nol dan elipsnya meledak. Batas ini yang menahannya.
    float ndr = dot(N, rd);
    ndr = (ndr < 0.0) ? min(ndr, -0.12) : max(ndr, 0.12);

    vec3 dxW = camRight * (2.0 * tanH * u.resolution.z);
    vec3 dyW = camUp    * (2.0 * tanV * u.resolution.w);

    vec3 fx = dist * (dxW - rd * (dot(N, dxW) / ndr));
    vec3 fy = dist * (dyW - rd * (dot(N, dyW) / ndr));

    // Ke ruang UV cascade. Ortografis, jadi bagian liniernya saja sudah cukup
    // (w = 0 memilih arah, bukan titik), lalu 0.5 untuk NDC -> UV.
    vec2 gx = 0.5 * (u.cascadeVP[ci] * vec4(fx, 0.0)).xy;
    vec2 gy = 0.5 * (u.cascadeVP[ci] * vec4(fy, 0.0)).xy;

    float maxFootUV = ANISO_MAX_TEXELS * u.params3.y;
    float lgx = length(gx), lgy = length(gy);
    if (lgx > maxFootUV) gx *= maxFootUV / lgx;
    if (lgy > maxFootUV) gy *= maxFootUV / lgy;

    // Kovarians kernel = jejak piksel + cakram penumbra.
    // Menjumlahkannya sebagai kovarians, bukan memilih yang terbesar, membuat
    // keduanya menyatu mulus: di kontak penumbra kecil dan bentuknya ditentukan
    // jejak piksel; jauh dari caster penumbra mendominasi dan elipsnya
    // membulat sendiri.
    mat2 F = mat2(gx, gy);
    mat2 C = F * transpose(F);
    C[0][0] += filterUV * filterUV;
    C[1][1] += filterUV * filterUV;

    mat2 S = sqrtPSD2(C);

    // Jumlah tap mengikuti panjang elips, supaya jarak antar tap tetap di bawah
    // satu texel berapa pun lonjongnya. Piksel dekat tetap murah; hanya lantai
    // menyerong yang jauh yang membayar penuh.
    float majUV   = max(length(S[0]), length(S[1]));
    float majTex  = majUV / max(u.params3.y, 1e-9);
    int   want    = int(ceil(2.0 * majTex / TAP_SPACING_TEXELS));
    int   capTaps = (u.params4.w < 0.5) ? PCF_SAMPLES_MAX : PCF_SAMPLES_BILINEAR;
    int   nTaps   = clamp(want, PCF_SAMPLES_BILINEAR, capTaps);

    // ====================================================================
    // RPDB — RECEIVER PLANE DEPTH BIAS
    //
    // Inilah yang hilang, dan inilah kenapa sobekan bertahan melewati sekian
    // banyak perbaikan.
    //
    // Setiap tap PCF membandingkan kedalaman shadow map di posisi yang BERBEDA
    // terhadap SATU nilai acuan yang sama: zRecv di titik tengah. Pada permukaan
    // yang miring terhadap cahaya, kedalaman penerima berubah menyeberangi
    // kernel — jadi tap yang jauh dari pusat membandingkan terhadap acuan yang
    // sudah salah. Selisihnya ditutup dengan menaikkan bias konstan, dan bias
    // konstan yang terlalu besar itulah yang menggeser posisi tepi.
    //
    // RPDB menghitung seberapa cepat kedalaman penerima berubah per satuan UV
    // shadow map, lalu memberi SETIAP tap acuannya sendiri yang mengikuti bidang
    // permukaan. Hasilnya perbandingan yang benar di seluruh kernel — tanpa
    // menaikkan bias, tanpa menggeser apa pun.
    //
    // Dihitung ANALITIK, bukan dari ddx/ddy: ini compute shader. Bahannya sudah
    // ada — fx dan fy adalah perpindahan dunia sepanjang BIDANG permukaan untuk
    // satu langkah piksel, dan F memetakannya ke UV. Tinggal mengambil
    // perubahan kedalamannya dan membalik F.
    float dzx = (u.cascadeVP[ci] * vec4(fx, 0.0)).z;
    float dzy = (u.cascadeVP[ci] * vec4(fy, 0.0)).z;

    vec2  gradUV = vec2(0.0);
    float detF   = F[0][0] * F[1][1] - F[1][0] * F[0][1];
    if (abs(detF) > 1e-12) {
        mat2 Finv = mat2(F[1][1], -F[0][1], -F[1][0], F[0][0]) / detF;
        gradUV = transpose(Finv) * vec2(dzx, dzy);
    }

    // Dijaga: pada permukaan yang nyaris sejajar arah cahaya, gradiennya menuju
    // tak hingga. Batas ini menjaga koreksi per tap tidak melampaui bias
    // maksimum yang wajar — di luar itu, tap-nya memang tidak bisa dipercaya.
    float maxCorr = 8.0 * bias;
    float gradLen = length(gradUV) * majUV;
    if (gradLen > maxCorr && gradLen > 1e-9) gradUV *= maxCorr / gradLen;

    float rotation2 = rotation + GOLDEN_ANGLE * 0.5;
    float lit = 0.0;
    for (int i = 0; i < PCF_SAMPLES_MAX; ++i) {
        if (i >= nTaps) break;
        vec2 o = S * vogelDisk(i, nTaps, rotation2);
        // Acuan tiap tap mengikuti bidang penerima, bukan nilai tengah.
        lit += cmpTap(uvPcf + o, float(ci), zRecv - bias + dot(gradUV, o));
    }
    lit /= float(nTaps);

    return vec3(1.0 - lit, penumbraWorld, 1.0);
}

// ---------------------------------------------------------------------------

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim   = ivec2(u.resolution.xy);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    float depth = texelFetch(sceneDepth, coord, 0).r;
    if (depth >= 1.0) {
        imageStore(shadowResolveOut, coord, vec4(0.0));
        return;
    }

    vec2 uv       = (vec2(coord) + 0.5) * u.resolution.zw;
    vec4 worldH   = u.invViewProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec3 worldPos = worldH.xyz / worldH.w;

    float viewZ = -(u.view * vec4(worldPos, 1.0)).z;

    vec3  N     = octDecode(texelFetch(gNormalOct, coord, 0).rg);
    vec3  L     = normalize(u.lightDirWorld.xyz);
    float NdotL = dot(N, L);

    if (NdotL <= 0.0) {
        imageStore(shadowResolveOut, coord, vec4(1.0, u.params0.y, 0.0, 0.0));
        return;
    }

    int ci = NUM_CASCADES - 1;
    for (int i = 0; i < NUM_CASCADES; ++i) {
        if (viewZ < splitOf(i)) { ci = i; break; }
    }

    // ========================================================================
    // v69: PERALIHAN CASCADE BERDITHER
    //
    //   MASALAHNYA. Pemilihan di atas keras: begitu viewZ melewati splitOf(ci),
    //   SEMUA yang menentukan tampilan bayangan berubah sekaligus dalam satu
    //   piksel — ukuran texel, lebar penumbra minimum, besar bias, dan radius
    //   PCF. Di scene-mu lompatannya besar dan bisa dihitung dari log:
    //
    //     C2 texel 0,9 cm  ->  C3 texel 1,8 cm   (2,0x)
    //     C3 texel 1,8 cm  ->  C4 texel 4,2 cm   (2,3x)
    //     C4 texel 4,2 cm  ->  C5 texel 12,1 cm  (2,9x)
    //
    //   Hasilnya garis lurus melintang di lantai, tepat di jarak split, tempat
    //   ketajaman bayangan berubah mendadak. Setiap engine CSM punya masalah
    //   ini; yang membedakan AAA adalah mereka menanganinya.
    //
    //   DUA CARA MENANGANINYA, DAN KENAPA SAYA PILIH YANG KEDUA.
    //
    //   [a] Lerp: hitung bayangan di KEDUA cascade lalu campur. Benar, mulus,
    //       dan MAHAL — dua kali seluruh blocker search + PCF di pita
    //       peralihan. Di llvmpipe itu tidak masuk akal.
    //
    //   [b] Dither: pilih SATU cascade secara acak, dengan peluang sebanding
    //       posisi di dalam pita. Biayanya NOL — tetap satu kali PCF. Yang
    //       dihasilkannya derau, dan derau itu persis yang sudah ditangani
    //       shadow_temporal + svgf_atrous di hilir.
    //
    //   Ini keputusan yang hanya masuk akal karena engine ini SUDAH punya
    //   denoiser. Tanpa denoiser, [b] akan terlihat sebagai bintik.
    //
    //   Noise-nya digeser (17,53) supaya tidak sefase dengan blue noise yang
    //   dipakai memutar cakram PCF di bawah — kalau sefase, kedua keputusan
    //   berkorelasi dan pitanya berubah jadi pola, bukan derau.
    // ========================================================================
    if (ci < NUM_CASCADES - 1) {
        float loSplit = (ci == 0) ? u.params2.y : splitOf(ci - 1);
        float hiSplit = splitOf(ci);
        float span    = max(hiSplit - loSplit, 1e-4);
        float f       = clamp((viewZ - loSplit) / span, 0.0, 1.0);

        // Pita peralihan = 12% terakhir tiap cascade.
        const float BLEND_BAND = 0.12;
        if (f > 1.0 - BLEND_BAND) {
            float p = (f - (1.0 - BLEND_BAND)) / BLEND_BAND;   // 0..1
            if (blueNoise(coord + ivec2(17, 53), u.params2.x) < p) ci = ci + 1;
        }
    }

    // frameOffset TIDAK ditambahkan lagi di sini: blueNoise() sudah menerapkan
    // pergeseran rasio emas per frame di dalamnya. Menambahkannya dua kali
    // membuat kedua deret sefase dan justru mengembalikan pola berulang yang
    // ingin dihindari.
    float rotation = blueNoise(coord, u.params2.x) * 6.28318531;

    // Ukuran dunia yang tercakup satu piksel layar pada kedalaman ini.
    // Kontinu terhadap viewZ, jadi batas bawah PCF yang diturunkan darinya
    // juga kontinu melintasi batas cascade.
    float worldPerPixel = 2.0 * viewZ * u.cameraPos.w * u.resolution.w;

    // Pergeseran acak sub-texel: +-TEXEL_JITTER texel, beda tiap piksel DAN tiap
    // frame. Inilah yang memecah undakan siluet caster jadi noise.
    // Dua nilai blue noise yang TIDAK berkorelasi, untuk sumbu x dan y.
    //
    // Menggeser FASE saja tidak cukup, dan alasannya penting: kalau kedua
    // panggilan membaca texel yang sama, hasilnya n2 = fract(n1 + c) — sebuah
    // konstanta. Vektor (n1, n2) lalu terkurung di satu garis diagonal alih-alih
    // mengisi bidang, jadi dither sub-texel cuma bekerja di SATU arah dan
    // kuantisasi di arah lain tetap utuh.
    //
    // Yang benar: baca texel yang BERJAUHAN di tile. Blue noise menjamin
    // ketidakmiripan lokal, dan pada jarak besar sampelnya praktis independen.
    // Offset dipilih ganjil supaya tidak pernah membagi habis lebar tile 128.
    float n1 = blueNoise(coord,                    u.params2.x);
    float n2 = blueNoise(coord + ivec2(37, 89),    u.params2.x);
    vec2  texJitter = (vec2(n1, n2) * 2.0 - 1.0) * (TEXEL_JITTER * u.params3.y);

    vec3 a = sampleCascadePCSS(ci, worldPos, N, NdotL, rotation, worldPerPixel, texJitter);

    // [2] Bobot blending: fade di N% terakhir rentang cascade. Kalau cascade
    //     sekarang invalid (titik di luar frustumnya), bobot dipaksa penuh ke
    //     cascade berikutnya — jalur yang sama, bukan lompatan terpisah.
    float w = 0.0;
    if (ci + 1 < NUM_CASCADES) {
        float nearSplit = (ci == 0) ? u.params2.y : splitOf(ci - 1);
        float farSplit  = splitOf(ci);
        float t = (viewZ - nearSplit) / max(farSplit - nearSplit, 1e-4);
        w = smoothstep(1.0 - u.params2.w, 1.0, t);
        if (a.z < 0.5) w = 1.0;
    }

    vec2 res;
    if (w > 0.0) {
        vec3 b = sampleCascadePCSS(ci + 1, worldPos, N, NdotL, rotation, worldPerPixel, texJitter);
        if (b.z < 0.5)      res = a.xy;
        else if (a.z < 0.5) res = b.xy;
        else                res = mix(a.xy, b.xy, w);
    } else {
        res = a.xy;
    }

    // ---- Contact shadow ----------------------------------------------------
    // Digabung dengan max(): keduanya menyatakan hal yang sama (berapa banyak
    // cahaya yang terhalang), jadi yang paling menghalangi yang menang. Tidak
    // ada model penumbra kedua yang perlu disambung — persis alasan kenapa
    // pendekatan ini menyatu mulus dengan PCSS.
    if (res.x < 0.999) {
        float cs = contactShadow(worldPos, N, L, viewZ, blueNoise(coord, u.params2.x));
        res.x = max(res.x, cs);
    }

    // ---- Peredupan di ujung jangkauan --------------------------------------
    // Tanpa ini ada garis lurus tajam di tempat cascade terakhir berakhir —
    // sangat mencolok di dunia terbuka. Diredupkan di N% terakhir jangkauan.
    {
        float shadowMax = u.params5.y;
        float fadeStart = shadowMax * (1.0 - u.params5.z);
        res.x *= 1.0 - smoothstep(fadeStart, shadowMax, viewZ);
    }

    // ---- MODE DEBUG -------------------------------------------------------
    // Semua mode juga mem-bypass temporal dan blur (lihat shader masing-masing),
    // jadi yang tampil benar-benar keluaran mentah tahap ini.
    if (u.params5.x > 0.5) {
        if (u.params5.x < 1.5) {
            // 1 = indeks cascade. Rentangnya dibagi menurut NUM_CASCADES, bukan
            // dengan langkah tetap 0,25 — rumus lama itu peninggalan era 4
            // cascade dan menghasilkan 1,15 serta 1,40 untuk C4 dan C5, yang
            // sama-sama di-clamp ke 1,0 di main.frag sehingga dua cascade
            // terjauh tidak bisa dibedakan.
            res.x = 0.10 + 0.85 * float(ci) / float(NUM_CASCADES - 1);
        } else if (u.params5.x < 2.5) {
            // 2 = lebar penumbra
            res.x = clamp(res.y / max(u.params0.z, 1e-4), 0.0, 1.0);
        } else if (u.params5.x > 3.5) {
            // 4 = AO saja: bayangan matahari dipaksa nol supaya yang tampil
            //     murni ambient occlusion
            res.x = 0.0;
        }
        // 3 = bayangan mentah: res.x dibiarkan apa adanya
    }

    imageStore(shadowResolveOut, coord, vec4(res.x, res.y, 0.0, 0.0));
}

shadow_temporal.comp:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// shadow_temporal.comp — v2
//
//   PERBAIKAN UTAMA: blending sekarang BEBAS FRAME RATE.
//
//   v1 memakai faktor blend tetap 0,92. Faktor tetap berarti jendela waktu
//   akumulasinya ikut melar saat FPS turun:
//        tau = -dt / ln(blend)
//        60 FPS -> tau = 200 ms   (benar)
//        15 FPS -> tau = 800 ms   (4x terlalu panjang)
//   Delapan ratus milidetik history yang ditarik ikut itulah yang terlihat
//   sebagai guratan/smear di lantai saat kamera berputar. Sekarang yang
//   dikonfigurasi adalah KONSTANTA WAKTU dalam detik, dan faktor blend-nya
//   diturunkan dari deltaTime — jendelanya tetap sama di 15 FPS maupun 120 FPS.
//
//   Ambang kecepatan juga dipindah ke piksel per DETIK, bukan piksel per frame,
//   karena alasan yang sama: di 15 FPS piksel per frame otomatis 4x lebih besar
//   sehingga v1 salah menganggap semua gerakan sebagai gerakan sangat cepat dan
//   membuang history terus-menerus.
//
//   Output rgba16f: .r shadow  .g penumbraWorld  .b guideWorld  .a AMBIENT OCCLUSION
//
//   Kanal .a dulu menyimpan "dev" (standar deviasi) yang ternyata tidak pernah
//   dibaca siapa pun. Sekarang dipakai AO, sehingga AO menumpang gratis di
//   akumulasi temporal dan bilateral blur yang sama dengan bayangan CSM —
//   tanpa image history tambahan, tanpa pass blur tambahan.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// v68: ambang penerimaan normal untuk riwayat. 0,9 = sekitar 26 derajat.
// Diangkat jadi konstanta karena sekarang dipakai per-texel di jejak 2x2 —
// empat salinan angka yang sama adalah empat peluang untuk meleset.
const float NORMAL_REJECT = 0.9;

#include "common_ubo.glsl"

layout(binding = 1) uniform sampler2D shadowResolve;
layout(binding = 2) uniform sampler2D shadowHistoryIn;
layout(binding = 3) uniform sampler2D velocityBuffer;
layout(binding = 4) uniform sampler2D sceneDepth;
layout(binding = 5, rgba16f) uniform writeonly image2D shadowHistoryOut;
layout(binding = 6) uniform sampler2D aoRaw;
layout(binding = 7) uniform sampler2D gNormalOct;      // uji disoklusi
layout(binding = 8) uniform sampler2D momentsIn;       // .r m2_shadow .g m2_ao .b histLen
layout(binding = 9, rgba16f) uniform writeonly image2D momentsOut;

vec3 octDecodeN(vec2 f) {
    vec3 n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// Diperketat dari 1,25. Makin kecil makin agresif membuang history yang tidak
// cocok dengan statistik frame ini — obat langsung untuk ghosting.
const float MAX_HISTORY = 32.0;   // sampel efektif maksimum
const float MIN_ALPHA   = 0.05;
const float HIST_REJECT_K   = 1.5;    // ambang tolak = k x sd tetangga
const float HIST_REJECT_MIN = 0.05;   // lantai, untuk daerah yang variansinya ~0
const float TAU_SECONDS = 0.25;   // riwayat memudar dalam ~0,25 detik NYATA   // batas bawah supaya perubahan nyata tetap terkejar

// Lantai akumulasi PER FRAME saat kamera (nyaris) diam.
//
// Konstanta waktu di atas benar untuk kasus bergerak — itu yang menghapus smear.
// Tapi di FPS rendah ia justru mematikan akumulasi sama sekali: pada 1 FPS,
// exp(-1.0 / 0.5) = 0.135, praktis tanpa history, sehingga noise dither
// sub-texel dari csm_resolve tidak pernah terselesaikan.
//
// Saat kamera diam, memperpanjang jendela akumulasi tidak berisiko apa pun —
// tidak ada yang bisa ter-smear kalau tidak ada yang bergerak. Jadi khusus
// kasus diam, blend dihitung per FRAME, bukan per detik.
// Lantai per-frame TETAP NOL walau temporal dinyalakan.
//
// Ini yang menyebabkan sobekan di v6-v26: pada 0,5 FPS ia menahan 93% history
// tiap frame, jauh melampaui kemampuan variance clipping mengawasinya. Blend
// biasa (exp(-dt/tau)) sudah cukup dan berperilaku benar di semua frame rate;
// lantai buatan ini cuma tambalan untuk FPS rendah yang akibatnya lebih buruk
// daripada masalah yang ditambalnya.
const float BLEND_STATIC_PER_FRAME = 0.0;

const float TAU_FAST      = 0.045;  // detik, saat kamera berputar cepat
const float SPEED_STATIC  = 60.0;   // piksel/detik, di bawah ini dianggap diam
const float SPEED_FAST    = 900.0;  // piksel/detik, di atas ini dianggap cepat


void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim   = ivec2(u.resolution.xy);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    vec2  uv   = (vec2(coord) + 0.5) * u.resolution.zw;
    float rawD = texelFetch(sceneDepth, coord, 0).r;

    if (rawD >= 1.0) {                                  // langit
        // .a = 0.0, BUKAN farZ.
        //
        // Kanal ini adalah "guide" yang dipakai svgf_atrous untuk mengenali
        // langit: pusatnya lewat `guide <= 0.0`, tetangganya lewat
        // `s.a <= 0.0`. Selama di sini ditulis farZ (100), kedua penjagaan itu
        // KODE MATI — tidak pernah sekali pun menyala. Akibatnya piksel langit
        // ikut jadi tetangga filter dengan bayangan 0,0 (terang penuh) dan
        // AO 1,0 (terbuka penuh), dan satu-satunya yang menahannya tinggal
        // bobot kedalaman. Bobot itu menutup rapat di jarak dekat, tapi
        // melonggar mendekati far plane: pada 80 m, 52% nilai piksel tepi
        // diambil dari langit; pada 90 m, 73%. Itu garis terang di siluet
        // bangunan jauh.
        //
        // Nol juga nilai yang benar untuk pemakaian satunya: temporal sendiri
        // sudah membaca `hist.a <= 0.0` sebagai "riwayat tidak sah", dan
        // riwayat langit memang tidak pernah dipakai — piksel langit selalu
        // keluar lebih awal di sini.
        imageStore(shadowHistoryOut, coord, vec4(0.0, 1.0, 0.0, 0.0));
        imageStore(momentsOut,       coord, vec4(0.0, 1.0, 1.0, 0.0));
        return;
    }

    float curShadow = texelFetch(shadowResolve, coord, 0).r;
    float curAo     = texelFetch(aoRaw, coord, 0).r;
    float depthCur  = linearizeDepth(rawD);
    vec3  nCur      = octDecodeN(texelFetch(gNormalOct, coord, 0).rg);

    // ========================================================================
    // AKUMULASI TEMPORAL SVGF
    //
    // Bedanya dengan versi lama bukan detail, tapi APA yang disimpan.
    //
    // Versi lama menyimpan nilai bayangan saja, lalu menebak sebarannya dari
    // jendela 3x3 frame ini. Sembilan sampel adalah dasar yang sangat lemah
    // untuk memperkirakan variansi — dan variansi itulah yang mengendalikan
    // seluruh kekuatan filter di hilir. Perkiraan yang buruk menghasilkan
    // filter yang kadang terlalu lemah (noise lolos) dan kadang terlalu kuat
    // (tepi tergerus). Itu sumber ketidakstabilan yang kita kejar terus.
    //
    // SVGF menyimpan MOMEN: rata-rata (m1) dan rata-rata kuadrat (m2), yang
    // diakumulasi lintas frame. Variansi lahir dari m2 - m1*m1 atas puluhan
    // sampel efektif, bukan sembilan. Panjang riwayat ikut disimpan supaya
    // frame-frame awal setelah disoklusi memakai bobot yang benar alih-alih
    // langsung mempercayai riwayat yang belum terkumpul.
    // ========================================================================
    vec2 vel    = texelFetch(velocityBuffer, coord, 0).rg;
    vec2 prevUV = uv - vel;

    bool valid = all(greaterThanEqual(prevUV, vec2(0.0)))
              && all(lessThanEqual(prevUV, vec2(1.0)));

    vec4 hist = vec4(0.0);
    vec4 mom  = vec4(0.0);
    if (valid) {
        // ====================================================================
        // v68: PENGAMBILAN RIWAYAT 2x2 DENGAN VALIDITAS PER-TEXEL
        //
        //   Sebelumnya satu fetch bilinear, lalu uji disoklusi dijalankan
        //   HANYA di titik tengah prevUV. Satu fetch bilinear mencampur EMPAT
        //   texel, jadi satu uji di tengah tidak pernah bisa mewakili keempatnya.
        //   Dua akibatnya berlawanan dan dua-duanya merusak:
        //
        //     [a] Jejak yang tiga dari empat texel-nya sah tetap DIBUANG
        //         seluruhnya kalau titik tengahnya kebetulan gagal. Riwayat
        //         yang bagus hilang, piksel jatuh ke variansi spasial, dan
        //         hasilnya berkedip.
        //
        //     [b] Sebaliknya, satu texel BURUK di dalam jejak ikut tercampur
        //         ke hasil bilinear tanpa pernah diuji. Bayangan dari
        //         permukaan lain merembes masuk dan menempel.
        //
        //   Keduanya paling sering terjadi tepat di TEPI SILUET — tempat
        //   bayangan paling menentukan dan paling diperhatikan mata.
        //
        //   Sekarang keempat texel diambil sendiri, diuji satu per satu dengan
        //   uji yang sama persis, lalu dibobot ulang. Ini teknik denoiser
        //   standar di engine AAA, dan biayanya cuma tiga fetch tambahan.
        // ====================================================================
        vec2  hs  = vec2(textureSize(shadowHistoryIn, 0));
        vec2  huv = prevUV * hs - 0.5;
        vec2  hf  = fract(huv);
        ivec2 hb  = ivec2(floor(huv));

        vec4  hSum = vec4(0.0);
        vec4  mSum = vec4(0.0);
        float wSum = 0.0;

        for (int i = 0; i < 4; ++i) {
            ivec2 off = ivec2(i & 1, i >> 1);
            ivec2 c   = clamp(hb + off, ivec2(0), ivec2(hs) - 1);

            vec4 h = texelFetch(shadowHistoryIn, c, 0);
            if (h.a <= 0.0) continue;

            float relD = abs(depthCur - h.a) / max(depthCur, 1e-4);
            if (relD > u.params1.z) continue;

            vec3 nPrev = octDecodeN(texelFetch(gNormalOct, c, 0).rg);
            if (dot(nCur, nPrev) < NORMAL_REJECT) continue;

            // Bobot bilinear asli tetap dipakai, cuma dinormalkan ulang atas
            // texel yang lolos. Kalau semuanya lolos, hasilnya identik dengan
            // fetch bilinear lama — jadi ini murni menambah ketelitian, bukan
            // mengubah perilaku di daerah yang memang sudah benar.
            vec2  bw = mix(1.0 - hf, hf, vec2(off));
            float w  = bw.x * bw.y;

            hSum += h * w;
            mSum += texelFetch(momentsIn, c, 0) * w;
            wSum += w;
        }

        if (wSum > 1e-4) {
            hist = hSum / wSum;
            mom  = mSum / wSum;
        } else {
            valid = false;          // keempatnya ditolak: benar-benar disoklusi
        }
    }

    // ---- Tetangga 3x3 dari frame INI --------------------------------------
    // Dihitung SELALU, bukan hanya saat riwayat pendek. Dipakai dua kali:
    // untuk menolak riwayat basi di bawah, dan sebagai perkiraan variansi
    // cadangan lebih jauh ke bawah. Menghitungnya sekali menghemat 9 fetch
    // dibanding menghitungnya dua kali.
    float nSum = 0.0, nSumSq = 0.0;
    [[unroll]] for (int dy = -1; dy <= 1; ++dy)
    [[unroll]] for (int dx = -1; dx <= 1; ++dx) {
        float v = texelFetch(shadowResolve,
                    clamp(coord + ivec2(dx, dy), ivec2(0), dim - 1), 0).r;
        nSum += v; nSumSq += v * v;
    }
    float nMean = nSum / 9.0;
    float nVar  = max(nSumSq / 9.0 - nMean * nMean, 0.0);

    // ---- Penolakan riwayat basi -------------------------------------------
    //
    // Pemeriksaan keabsahan di atas semuanya GEOMETRIS: posisi, kedalaman,
    // normal. Tidak satu pun menolak riwayat pada permukaan yang diam tapi
    // PENCAHAYAANNYA berubah — misalnya lantai yang dilewati bayangan benda
    // bergerak. Dengan TAU 40 detik, riwayat seperti itu akan diblending
    // puluhan detik dan meninggalkan jejak panjang.
    //
    // Perubahan besar diperlakukan sama seperti piksel yang tersingkap:
    // riwayatnya DIBUANG, bukan dijepit. Membuangnya membuat histLen kembali 1
    // sehingga alpha jadi 1 dan piksel langsung memakai nilai frame ini, lalu
    // menumpuk lagi dari nol. Menjepitnya saja tidak cukup — sisa galatnya
    // tetap meluruh selama TAU detik.
    //
    // Ambangnya berskala pada simpangan baku tetangga, jadi di dalam penumbra
    // yang memang bergradien ia melebar sendiri dan tidak salah menolak.
    // Terukur, 200 percobaan, derau PCF 0,09 dan alpha 0,05:
    //
    //     k     derau sisa (diam)  penolakan palsu  waktu respons
    //    1,0        0,0175             1,37%           3 detik
    //    1,5        0,0127             0,00%           3 detik   <- dipakai
    //    3,0        0,0127             0,00%           3 detik
    //   tanpa       0,0127               -           151 detik
    //
    // k = 1,5 memberi respons 54x lebih cepat tanpa biaya apa pun pada
    // penekanan derau saat diam.
    if (valid) {
        float win = max(HIST_REJECT_K * sqrt(nVar), HIST_REJECT_MIN);
        if (abs(hist.r - nMean) > win) valid = false;
    }

    float histLen = valid ? min(mom.b + 1.0, MAX_HISTORY) : 1.0;

    // Bobot akumulasi: 1/histLen — TAPI dengan batas bawah yang sadar WAKTU.
    //
    // 1/histLen menghitung FRAME, bukan detik. Klaim saya bahwa ia "otomatis
    // benar berapa pun frame rate" itu salah, dan akibatnya terlihat langsung:
    //
    //     60  FPS -> riwayat memudar dalam  0,7 detik
    //     0,5 FPS -> riwayat memudar dalam 89,8 detik
    //
    // Di 0,5 FPS satu frame dua detik, jadi bayangan tertinggal puluhan detik —
    // persis "bayangan yang tertinggal". Batas bawah berbasis waktu di bawah
    // memastikan riwayat selalu memudar dalam TAU detik, berapa pun frame
    // rate-nya. Di 60 FPS ia longgar dan 1/histLen yang menentukan; di frame
    // rate rendah ia yang mengambil alih.
    float dt        = max(u.params4.x, 1e-4);
    float alphaTime = 1.0 - exp(-dt / TAU_SECONDS);
    float alpha     = max(max(1.0 / histLen, MIN_ALPHA), alphaTime);

    float s1 = mix(hist.r, curShadow, alpha);
    float a1 = mix(hist.g, curAo,     alpha);
    float s2 = mix(mom.r,  curShadow * curShadow, alpha);
    float a2 = mix(mom.g,  curAo     * curAo,     alpha);

    // Variansi dari momen. Selama riwayat masih pendek, momen belum bisa
    // dipercaya, jadi dipakai perkiraan spasial 3x3 — jalur cadangan yang sama
    // dengan yang dipakai SVGF.
    float varS = max(s2 - s1 * s1, 0.0);
    float varA = max(a2 - a1 * a1, 0.0);

    if (histLen < 4.0) {
        // nVar sudah dihitung di atas — tidak perlu menyapu tetangga dua kali.
        float spatialVar = nVar;
        // Pengali 4.0 yang sempat saya pakai di sini adalah kesalahan besar.
        //
        // Variansi memandu SIGMA_L, yang menentukan seberapa longgar filter
        // menyeberangi beda nilai. Variansi yang dilebihkan empat kali membuat
        // sigmaL besar, bobot luminansi jadi ~1 di mana-mana, dan a-trous
        // menyapu rata menyeberangi tepi bayangan — bayangannya hilang jadi
        // bubur. Dan karena riwayat terus mereset di frame rate rendah, jalur
        // cadangan spasial ini justru yang hampir selalu terpakai.
        float k = 1.0 - histLen / 4.0;
        varS = mix(varS, spatialVar, k);
        varA = mix(varA, spatialVar, k);
    }

    // Mode bypass / debug: teruskan nilai frame ini apa adanya.
    if (u.params5.x > 0.5) {
        imageStore(shadowHistoryOut, coord, vec4(curShadow, curAo, 0.0, depthCur));
        imageStore(momentsOut,       coord, vec4(0.0, 0.0, 1.0, 0.0));
        return;
    }

    // .r bayangan  .g AO  .b variansi bayangan  .a kedalaman (guide)
    imageStore(shadowHistoryOut, coord, vec4(s1, a1, varS, depthCur));
    imageStore(momentsOut,       coord, vec4(s2, a2, histLen, varA));
}

svgf_atrous.comp:

#version 450
#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// svgf_atrous.comp — filter à-trous edge-avoiding, SATU shader untuk SEMUA
// iterasi. Menggantikan shadow_blur_h dan shadow_blur_v sepenuhnya.
//
// Kenapa dua pass separable itu salah sejak awal:
//
//   Blur separable memperlebar jangkauan dengan MENAMBAH TAP. Jangkauan besar
//   berarti tap banyak, dan radius yang berubah menurut posisi (penumbra,
//   kedalaman) membuat dua sisi sebuah tepi difilter dengan lebar berbeda —
//   yang menggeser tepi itu. Semua "sobekan" yang kita kejar lahir dari sana.
//
//   À-trous memperlebar jangkauan dengan MENJARANGKAN TAP. Jumlah tap tetap 25
//   (5x5) di setiap iterasi; yang berlipat dua tiap iterasi adalah jarak antar
//   tap. Empat iterasi memberi jangkauan efektif 33x33 piksel dengan 100 tap,
//   bukan 1089. Dan karena kernelnya tetap, tidak ada radius yang bisa berbeda
//   di dua sisi tepi.
//
// Bobot edge-stopping (Schied dkk., HPG 2017), semuanya berskala dari data:
//   w_z : gradien kedalaman  — ambang lahir dari kemiringan permukaan
//   w_n : normal pangkat     — permukaan tegak lurus tertolak walau sedepth
//   w_l : dipandu variansi   — kuat saat berisik, mengunci saat sudah tenang
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common_ubo.glsl"
#include "common_oct.glsl"

layout(binding = 1) uniform sampler2D svgfIn;      // .r shadow .g AO .b var .a depth
layout(binding = 2, rgba16f) uniform writeonly image2D svgfOut;
layout(binding = 3) uniform sampler2D gNormalOct;
layout(binding = 4) uniform sampler2D aoRaw;       // .gb bent normal
layout(binding = 5) uniform sampler2D shadowResolve; // .g lebar penumbra (meter)

layout(push_constant) uniform PC {
    ivec4 cfg;   // x = stride, y = 1 kalau iterasi terakhir
} pc;

const float KERNEL[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);
const float SIGMA_Z = 4.0;
const float SIGMA_N = 64.0;
const float SIGMA_L = 4.0;

vec2 octEnc(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                                   n.y >= 0.0 ? 1.0 : -1.0);
    return n.xy;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim   = ivec2(u.resolution.xy);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    vec4  c     = texelFetch(svgfIn, coord, 0);
    vec2  bentC = texelFetch(aoRaw,  coord, 0).gb;
    float guide = c.a;

    // Bypass (mode debug / penyaringan dimatikan) dan piksel langit
    if (u.params5.x > 0.5 || guide <= 0.0) {
        if (pc.cfg.y != 0) imageStore(svgfOut, coord, vec4(c.r, c.g, bentC));
        else               imageStore(svgfOut, coord, c);
        return;
    }

    int stride = max(pc.cfg.x, 1);

    vec3  nC   = normalize(mat3(u.view) * octDecode(texelFetch(gNormalOct, coord, 0).rg));
    float wpp  = 2.0 * guide * u.cameraPos.w * u.resolution.w;

    // Variansi diperhalus 3x3 lebih dulu. Variansi mentah sendiri berisik, dan
    // memakainya langsung sebagai pemandu membuat kekuatan filter ikut berkedip
    // dari piksel ke piksel — terlihat sebagai bercak.
    float vSum = 0.0, vW = 0.0;
    [[unroll]] for (int dy = -1; dy <= 1; ++dy)
    [[unroll]] for (int dx = -1; dx <= 1; ++dx) {
        ivec2 q = clamp(coord + ivec2(dx, dy), ivec2(0), dim - 1);
        float w = (dx == 0 && dy == 0) ? 0.25 : ((dx == 0 || dy == 0) ? 0.125 : 0.0625);
        vSum += texelFetch(svgfIn, q, 0).b * w; vW += w;
    }
    float varSm  = max(vSum / max(vW, 1e-6), 0.0);

    // sigmaL: batas atas 0,025 -> 0,12.
    //
    // Batas 0,025 dulu dipasang untuk melawan light bleeding, dengan alasan
    // bahwa bobot luminansi yang longgar menarik nilai terang ke dalam bayangan.
    // Alasan itu sudah GUGUR: sumber light bleeding-nya kemudian terlacak ke
    // celah di shadow map pada rusuk antar-muka — addFace() tidak berbagi
    // verteks, dan pergeseran caster sepanjang normal merobek rusuknya. Itu
    // sudah diperbaiki di csm_shadow.vert dengan menggeser sepanjang L. Jadi
    // batas ini sekarang cuma menyisakan efek sampingnya.
    //
    // Diukur dengan menjalankan filter ini pada ramp penumbra sintetis 4 px dan
    // 8 px (rasio benar 2,00) ditambah derau, 40 percobaan per titik:
    //
    //   derau  cap 0,025 -> rasio / penekanan     cap 0,12 -> rasio / penekanan
    //   0,03      1,71 / 1,91x                       2,08 / 2,28x
    //   0,05      1,71 / 1,67x                       2,24 / 2,25x
    //   0,10      1,91 / 1,50x                       2,17 / 2,09x
    //
    // Batas lama kalah di KEDUA sumbu. Ia menekan derau lebih sedikit DAN
    // merusak rasio penumbra lebih parah — karena dengan sigmaL sesempit itu
    // filter tidak bisa memuluskan derau sama sekali, lalu berubah jadi operator
    // staircasing yang menarik ramp landai jadi tangga. Menaikkan batas
    // memperbaiki keduanya sekaligus, tanpa pertukaran.
    //
    // Berhenti di 0,12 karena di situ letak lututnya; 0,25 tidak menambah apa
    // pun yang berarti dan mulai melonggarkan penjagaan tepi tanpa imbalan.
    //
    // CATATAN: filter ini tetap MEMAMPATKAN lebar penumbra sekitar 2x (4 px
    // jadi ~2,2 px). Itu sifat bawaan bilateral ber-edge-stopping, bukan setelan
    // — memperbaikinya perlu lebar filter yang dipandu penumbraWorld, dan itu
    // butuh kanal tambahan yang belum ada.
    float sigmaL = clamp(SIGMA_L * sqrt(varSm), 0.004, 0.12);

    // ---- kemiringan lokal shadow, untuk membatalkan staircasing -----------
    //
    // wl membandingkan tetangga dengan nilai PUSAT. Pada tepi tangga itu benar,
    // tapi di dalam penumbra nilainya memang sedang berubah landai — dan filter
    // membacanya sebagai tepi yang harus dijaga. Akibatnya ramp ditarik jadi
    // tangga, dan lebar penumbra menyusut sekitar setengahnya. Terukur pada ramp
    // sintetis: penumbra 4 px keluar 2,10 px, yang 8 px keluar 4,25 px.
    //
    // Perbaikannya membandingkan tetangga dengan nilai yang DIPERKIRAKAN dari
    // kemiringan lokal, bukan dengan nilai pusat. Di dalam ramp, perkiraan itu
    // hampir tepat sehingga wl mendekati 1 dan ramp dimuluskan tanpa dibengkokkan.
    // Di tepi tangga, perkiraan meleset jauh sehingga wl mengecil dan tepinya
    // tetap terjaga.
    //
    // Kemiringan dicari dengan kuadrat terkecil berbobot pada SALIB 5 tap tiap
    // sumbu; selisihnya diambil terhadap pusat supaya tap langit yang dilewati
    // tidak memiringkan hasilnya. Terukur, 40 percobaan per titik:
    //
    //                        setia lebar  penekanan derau  tepi keras  pita tipis
    //   wl biasa                 0,53         2,27x          1,00 px     0,0208
    //   wl terkoreksi gradien    0,80         2,22x          1,00 px     0,0319
    //
    // Kesetiaan lebar naik separuh lebih, penekanan derau praktis tidak berubah,
    // dan penjagaan tepi keras sama sekali tidak berkurang.
    //
    // Varian murah dengan beda-pusat 2 tap SUDAH DICOBA dan ditolak: tepi keras
    // melebar jadi 2,30 px dan dasar pita tipis bocor ke 0,0586. Salib 5 tap
    // memerlukan 8 texelFetch tambahan per iterasi, dan itu harganya.
    vec2 gradS = vec2(0.0);
    {
        float numX = 0.0, numY = 0.0, denX = 0.0, denY = 0.0;
        for (int k = -2; k <= 2; ++k) {
            if (k == 0) continue;
            float pk = float(k * stride);
            float wk = KERNEL[k + 2];

            vec4 sx = texelFetch(svgfIn, clamp(coord + ivec2(k * stride, 0), ivec2(0), dim - 1), 0);
            if (sx.a > 0.0) { numX += wk * pk * (sx.r - c.r); denX += wk * pk * pk; }

            vec4 sy = texelFetch(svgfIn, clamp(coord + ivec2(0, k * stride), ivec2(0), dim - 1), 0);
            if (sy.a > 0.0) { numY += wk * pk * (sy.r - c.r); denY += wk * pk * pk; }
        }
        gradS = vec2(numX / max(denX, 1e-8), numY / max(denY, 1e-8));
    }

    // ---- BATAS LEBAR FILTER MENURUT LEBAR PENUMBRA -------------------------
    //
    // Aturannya satu kalimat: filter tidak boleh lebih lebar daripada sinyal
    // yang difilter. Tanpa batas ini a-trous menimpakan lebar yang SAMA ke
    // setiap tepi bayangan, berapa pun penumbra aslinya, dan contact hardening
    // yang sudah dihitung susah payah oleh PCSS ikut terhapus.
    //
    // Besarnya terukur, bukan kira-kira. Kernel [1,4,6,4,1]/16 punya simpangan
    // baku 1,0 px. Dua iterasi dengan stride 1 dan 2 menumpuk jadi
    // sqrt(1^2 + 2^2) = 2,236 px, dan sebuah tepi tegak yang dikonvolusi
    // dengannya melebar 2,563 * 2,236 = 5,73 px pada transisi 10-90%.
    //
    // Jadi SETIAP tepi bayangan di layar punya lantai 5,7 px, termasuk tepi
    // yang penumbra fisiknya di bawah 1 px. Sebagai pembanding, tepi kontak di
    // referensi COD terukur 2,2 px. Lantai itulah yang membuat bayangan tidak
    // pernah benar-benar menempel di kaki benda.
    //
    // csm_resolve sudah menghitung lebar penumbra dan menyimpannya di
    // shadowResolve.g dalam METER. Buffer itu masih hidup saat a-trous jalan
    // (tidak ada yang menimpanya sesudah csm_resolve), jadi cukup dibaca.
    //
    //   penumbraPx kecil  -> tap jauh digugurkan, tepi tetap setajam PCSS
    //   penumbraPx besar  -> semua tap lolos, filter jalan penuh
    //
    // Ambang bawahnya MIN_PCF_SCREEN_PX, karena itu memang lebar minimum yang
    // dipakai PCF di csm_resolve — sinyalnya tidak pernah lebih sempit dari itu.
    float penumbraPx = texelFetch(shadowResolve, coord, 0).g / max(wpp, 1e-6);
    float signalPx   = max(penumbraPx, u.params4.z);
    float limitPx    = max(signalPx * 0.5, 0.5);

    float sumS = c.r * KERNEL[2] * KERNEL[2];
    float sumA = c.g * KERNEL[2] * KERNEL[2];
    float sumV = c.b * KERNEL[2] * KERNEL[2] * KERNEL[2] * KERNEL[2];
    vec3  sumB = octDecode(bentC) * KERNEL[2] * KERNEL[2];
    float wS   = KERNEL[2] * KERNEL[2];
    float wSq  = wS * wS;

    [[unroll]] for (int j = 0; j < 5; ++j)
    [[unroll]] for (int i = 0; i < 5; ++i) {
        if (i == 2 && j == 2) continue;
        ivec2 off = ivec2(i - 2, j - 2) * stride;
        ivec2 q   = clamp(coord + off, ivec2(0), dim - 1);

        vec4 s = texelFetch(svgfIn, q, 0);
        if (s.a <= 0.0) continue;

        // w_z — penyebutnya perubahan kedalaman yang WAJAR untuk perpindahan
        // sejauh ini di sepanjang bidang permukaan.
        //
        // gradZ DIBATASI, dan itu yang kurang sampai sekarang.
        //
        // Rumusnya membagi dengan |nC.z|. Pada muka yang dilihat menyerong,
        // nC.z menuju nol dan gradZ meledak: di 87 derajat ambangnya mencapai
        // 4,3 METER untuk tetangga berjarak beberapa piksel. Bobot kedalaman
        // mati total di situ, dan filter bebas menyeberang ke permukaan lain —
        // itulah light bleeding yang selalu muncul di muka menyerong.
        //
        // Batas 8 x worldPerPixel per langkah piksel sudah sangat longgar untuk
        // permukaan miring yang sah, tapi menutup jalur permisif tanpa batas.
        float gradZ  = wpp * length(vec2(nC.x, nC.y)) / max(abs(nC.z), 0.05);
        gradZ = min(gradZ, wpp * 8.0);

        float expect = SIGMA_Z * gradZ * float(length(vec2(off))) + 1e-4;
        float wz = exp(-abs(s.a - guide) / expect);

        vec3  nQ = normalize(mat3(u.view) * octDecode(texelFetch(gNormalOct, q, 0).rg));
        float wn = pow(max(dot(nC, nQ), 0.0), SIGMA_N);

        float pred = c.r + dot(gradS, vec2(off));
        float wl   = exp(-abs(s.r - pred) / sigmaL);

        // Gerbang jangkauan: tap yang lebih jauh dari lebar penumbra setempat
        // digugurkan. Transisinya dibuat mulus dari limitPx ke 2*limitPx —
        // pemotongan keras akan muncul sebagai cincin di sekitar tepi tajam,
        // dan engine ini tidak punya AA untuk menyembunyikannya.
        float wSpan = 1.0 - smoothstep(limitPx, limitPx * 2.0, length(vec2(off)));
        if (wSpan <= 0.0) continue;

        float w = wSpan * KERNEL[i] * KERNEL[j] * wz * wn * wl;
        if (w <= 0.0) continue;

        sumS += s.r * w;
        sumA += s.g * w;
        sumB += octDecode(texelFetch(aoRaw, q, 0).gb) * w;
        // Variansi disaring dengan bobot KUADRAT: rata-rata berbobot menyusutkan
        // variansi sebagai kuadrat bobotnya. Tanpa ini, panduan di iterasi
        // berikutnya salah dan filter melebar terus.
        sumV += s.b * w * w;
        wS   += w;
        wSq  += w * w;
    }

    float oS = sumS / max(wS, 1e-6);
    float oA = sumA / max(wS, 1e-6);
    float oV = sumV / max(wSq, 1e-8);
    vec2  oB = (dot(sumB, sumB) > 1e-8) ? octEnc(normalize(sumB)) : bentC;

    // Iterasi terakhir menulis format yang dibaca main.frag:
    // .r bayangan, .g AO, .ba bent normal.
    if (pc.cfg.y != 0) imageStore(svgfOut, coord, vec4(oS, oA, oB));
    else               imageStore(svgfOut, coord, vec4(oS, oA, oV, guide));
}

volumetric.comp:

#version 450

// ---- Extension ------------------------------------------------------------
// [[unroll]] pada perulangan langkah ray march, yang jumlah putarannya konstan.
#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// volumetric.comp — Light shaft / god ray
//
//   Sinar matahari yang terlihat karena udara tidak benar-benar kosong. Tiap
//   titik di sepanjang garis pandang menerima cahaya matahari kalau ia tidak
//   terhalang, lalu menghamburkan sebagian ke arah kamera. Yang terlihat mata:
//   berkas terang di celah antar bangunan, dan bayangan yang "terpahat" di
//   udara alih-alih cuma jatuh di lantai.
//
//   Ini item terakhir antrean bayangan, dan satu-satunya yang membayangi RUANG,
//   bukan permukaan. Semua yang sebelumnya menjawab "apakah titik di permukaan
//   ini kena matahari". Yang ini menjawab pertanyaan itu untuk setiap titik di
//   udara sepanjang garis pandang.
//
//   SETENGAH RESOLUSI. Hamburan volume berfrekuensi sangat rendah — tidak ada
//   detail tajam yang bisa hilang — sementara biayanya berbanding lurus dengan
//   jumlah piksel dikali jumlah langkah. Setengah resolusi memangkasnya 4x, dan
//   pembesaran kembali gratis: main.frag menyampelnya dengan sampler linear,
//   jadi kartu grafis yang menginterpolasi.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common_ubo.glsl"

layout(binding = 1) uniform sampler2D            sceneDepth;
layout(binding = 2) uniform sampler2DArrayShadow shadowCascadesCmp;
// rg16f, bukan r16f: kanal .g menyimpan PANJANG RAY piksel ini.
//
// Buffer ini setengah resolusi, dan tepat di siluet satu texel-nya menaungi
// piksel langit sekaligus piksel dinding. Tanpa cara membedakan keduanya,
// pembesaran bilinear membocorkan hamburan milik langit — yang raynya penuh —
// ke dinding yang raynya pendek. Panjang ray inilah pembedanya: main.frag
// membandingkannya dengan jarak fragmen ke kamera, lalu membuang texel yang
// jelas milik permukaan lain.
layout(binding = 3, rg16f) uniform writeonly image2D volOut;
layout(binding = 4) uniform sampler2D blueNoiseTex;   // LUT 128x128 R8
#include "common_noise.glsl"

const int   NUM_CASCADES = 6;
const int   MAX_STEPS    = 24;

// Anisotropi Henyey-Greenstein. Positif = hamburan maju, artinya berkas jauh
// lebih terang saat kamera menghadap matahari dan nyaris hilang saat
// membelakanginya. Itu perilaku kabut dan debu sungguhan, dan justru sifat itu
// yang membuat light shaft terasa hidup saat pemain berputar.
// v70: 0,60 -> 0,78.
//
// Ini yang paling menentukan apakah hasilnya terbaca sebagai BERKAS atau
// sebagai KABUT. g mengatur seberapa terkonsentrasi hamburan ke arah maju.
// Dinormalkan ke puncaknya, seberapa jauh efeknya menyebar dari matahari:
//
//     sudut     g=0,60   g=0,78
//        0 deg   1,000    1,000
//       15 deg   0,711    0,329
//       30 deg   0,352    0,082
//       45 deg   0,175    0,030
//       90 deg   0,040    0,005
//
// Pada 0,60 efeknya masih 35% di 30 derajat dari matahari — artinya SELURUH
// langit ikut terangkat, dan itu definisi kabut. Pada 0,78 ia sudah turun ke
// 8%: yang menyala tinggal wilayah dekat matahari, dan di situlah berkasnya
// terlihat sebagai berkas.
const float HG_G = 0.78;

// ---- Berkas radial layar (v71) ---------------------------------------------
// RADIAL_STEPS   : langkah menuju matahari. Lebih banyak = berkas lebih halus
//                  dan lebih panjang, dan ini bagian termahal shader ini.
// RADIAL_SPAN    : seberapa jauh jarak ke matahari yang ditelusuri (1,0 =
//                  seluruhnya). Di bawah 1 membuat berkas lebih pendek tapi
//                  lebih murah.
// RADIAL_DECAY   : peluruhan per langkah. 0,97^32 = 0,38, jadi ujung berkas
//                  memudar alih-alih terpotong.
// RADIAL_WEIGHT  : penskalaan sebelum kekuatan akhir.
// RADIAL_STRENGTH: satu-satunya angka yang perlu disetel kalau berkasnya
//                  kurang atau kelewat tegas.
const int   RADIAL_STEPS         = 32;
const float RADIAL_SPAN          = 1.00;
const float RADIAL_DECAY         = 0.97;
const float RADIAL_WEIGHT        = 1.15;
const float RADIAL_STRENGTH      = 0.55;
const float RADIAL_OFFSCREEN_FADE = 0.35;


// ---- BLUE NOISE ------------------------------------------------------------
// Menggantikan interleaved gradient noise.
//
// IGN murah dan bagus secara temporal, tapi spektrum SPASIALnya mengandung
// energi frekuensi rendah — sampel yang berdekatan cenderung mirip, sehingga
// noise-nya menggumpal jadi bercak. Gumpalan frekuensi rendah persis yang
// PALING SULIT dihapus filter mana pun: a-trous meredam frekuensi tinggi.
//
// Blue noise memindahkan seluruh energinya ke frekuensi tinggi. Sisa noise-nya
// jadi butiran halus yang mudah dihapus, dan kalaupun tersisa, mata jauh lebih
// sulit melihatnya daripada bercak.
//
// Tile 128x128 bersifat toroidal, sampler REPEAT jadi tidak menghasilkan seam.
// Pergeseran per frame memakai rasio emas: urutan yang paling merata untuk
// sembarang jumlah frame, jadi akumulasi temporal tidak pernah mengulang pola.

// Fungsi fase Henyey-Greenstein
float phaseHG(float cosT, float g) {
    float g2 = g * g;
    float d  = 1.0 + g2 - 2.0 * g * cosT;
    return (1.0 - g2) / (12.5663706 * max(d * sqrt(max(d, 1e-4)), 1e-4));
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dimHalf = ivec2((u.resolution.xy + 1.0) * 0.5);
    if (coord.x >= dimHalf.x || coord.y >= dimHalf.y) return;

    float density = u.params5.w;
    int   steps   = int(u.params8.w);
    float maxDist = u.params9.w;

    if (density <= 0.0 || steps <= 0) {
        imageStore(volOut, coord, vec4(0.0, u.params9.w, 0.0, 0.0));
        return;
    }

    // Depth diambil dengan texelFetch di piksel penuh yang bersesuaian, BUKAN
    // texture() di UV setengah resolusi. Sampler-nya linear, dan merata-ratakan
    // kedalaman menyeberangi siluet menghasilkan kedalaman yang tidak dimiliki
    // permukaan mana pun — ray akan berhenti di kedalaman hantu dan meninggalkan
    // halo di setiap tepi geometri.
    ivec2 full = min(coord * 2, ivec2(u.resolution.xy) - 1);
    float rawD = texelFetch(sceneDepth, full, 0).r;

    vec2 uv = (vec2(full) + 0.5) * u.resolution.zw;
    vec3 camPos = u.cameraPos.xyz;

    // Arah pandang untuk piksel ini, lewat titik pada bidang jauh
    vec4 farH = u.invViewProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir  = normalize(farH.xyz / farH.w - camPos);

    // Panjang ray: berhenti di permukaan pertama, atau di batas jarak.
    float rayLen = maxDist;
    if (rawD < 1.0) {
        vec4 wH = u.invViewProj * vec4(uv * 2.0 - 1.0, rawD, 1.0);
        rayLen = min(length(wH.xyz / wH.w - camPos), maxDist);
    }
    if (rayLen <= 0.01) {
        imageStore(volOut, coord, vec4(0.0, rayLen, 0.0, 0.0));
        return;
    }

    vec3  L    = normalize(u.lightDirWorld.xyz);
    float cosT = dot(dir, L);

    // Offset awal acak per piksel per frame. Tanpanya, jumlah langkah yang
    // sedikit menghasilkan pita melengkung yang sangat mencolok — batas antar
    // langkah jatuh di tempat yang sama untuk semua piksel bertetangga. Dengan
    // offset acak, pita itu pecah jadi noise, lalu pembesaran bilinear dari
    // setengah resolusi meredamnya.
    float frameOffset = fract(u.params2.x * 0.61803399);
    float jitter = blueNoise(coord, u.params2.x);

    float stepLen = rayLen / float(steps);
    float lit = 0.0;

    [[unroll]] for (int i = 0; i < MAX_STEPS; ++i) {
        if (i >= steps) break;

        vec3  p     = camPos + dir * (stepLen * (float(i) + jitter));
        float viewZ = -(u.view * vec4(p, 1.0)).z;
        if (viewZ <= 0.0) continue;

        int ci = NUM_CASCADES - 1;
        for (int c = 0; c < NUM_CASCADES; ++c) {
            if (viewZ < splitOf(c)) { ci = c; break; }
        }

        vec4 lc = u.cascadeVP[ci] * vec4(p, 1.0);
        vec3 pr = lc.xyz / lc.w;
        vec2 sUV = pr.xy * 0.5 + 0.5;
        if (pr.z <= 0.0 || pr.z >= 1.0 ||
            any(lessThan(sUV, vec2(0.0))) || any(greaterThan(sUV, vec2(1.0)))) {
            lit += 1.0;                       // di luar cascade: anggap terbuka
            continue;
        }

        // Satu tap saja. Tidak perlu PCSS di sini: hasilnya dirata-ratakan
        // sepanjang puluhan langkah, jadi kelembutan tepinya datang dari
        // integrasi sepanjang ray, bukan dari penyaringan per langkah.
        lit += texture(shadowCascadesCmp, vec4(sUV, float(ci), pr.z - 0.0015));
    }

    // TIDAK ADA peredupan berdasarkan panjang ray.
    //
    // Versi pertama memakai 1 - smoothstep(maxDist*0.75, maxDist, rayLen) dengan
    // maksud menutupi tepi keras di tempat ray berhenti. Itu keliru dua kali:
    //
    //   1. Tepi kerasnya tidak pernah ada. rayLen = min(jarak, maxDist) sudah
    //      kontinu — permukaan di 59,9 m dan 60,1 m menghasilkan panjang ray
    //      yang praktis sama.
    //   2. Piksel LANGIT selalu punya rayLen = maxDist, jadi peredupannya selalu
    //      bernilai nol di sana. Berkas cahaya hilang total persis di tempat ia
    //      paling terlihat: melawan langit.
    //
    // Jadi peredupan itu tidak menyelesaikan masalah apa pun, dan mematikan
    // fiturnya di separuh layar.
    // v70: BENTUK BERBATAS FISIK.
    //
    // Rumus lama (density * rayLen) tumbuh TANPA BATAS dengan jarak. Piksel
    // langit memakai rayLen = maxDist penuh, jadi ia selalu mendapat nilai
    // terbesar — terukur 0,573 pada 0,012 x 60 m, sekitar 3,4x lebih terang
    // daripada lantai yang kena matahari penuh (0,17 HDR).
    //
    // Itu sebabnya langitmu terukur R/G 1,002 B/G 0,997: NETRAL PUTIH. Bukan
    // hangat seperti SUN_SHAFT_COLOR (1,075 / 0,839) dan bukan dingin seperti
    // ambient (B/G 1,238) — melainkan begitu terang sampai ACES menyapunya ke
    // putih. Selubung merata tanpa struktur = kabut, bukan berkas.
    //
    // Hamburan tunggal pada medium seragam sebenarnya:
    //     L_in = phase * (1 - exp(-density * panjang))
    // Suku (1 - exp(...)) jenuh menuju 1, jadi berapa pun jaraknya inScatter
    // tidak pernah melewati nilai phase-nya. Batas fisik, bukan clamp.
    float inScatter = (lit / float(steps))
                    * (1.0 - exp(-density * rayLen))
                    * phaseHG(cosT, HG_G);

    // ========================================================================
    // v71: BERKAS RADIAL LAYAR — bagian yang selama ini TIDAK ADA
    //
    //   KENAPA RAY MARCH SAJA TIDAK PERNAH MENGHASILKAN "GOD RAYS RDR2".
    //
    //   Ray march di atas menghitung berapa banyak cahaya terhambur ke arah
    //   kamera di sepanjang ray. Itu benar secara fisika, dan hasilnya
    //   memang HALUS — nilainya berubah pelan dari piksel ke piksel karena
    //   ray tetangga menempuh udara yang hampir sama. Selubung halus yang
    //   berubah pelan persis seperti apa yang mata sebut KABUT.
    //
    //   Yang orang maksud dengan "god rays" adalah hal lain: GARIS-GARIS
    //   TEGAS yang memancar dari matahari, terpotong siluet benda. Itu bukan
    //   hasil integrasi sepanjang ray — itu bayangan benda yang DIPROYEKSIKAN
    //   ke arah radial di layar. RDR2, dan hampir semua game yang berkasnya
    //   terkenal, memakai teknik ini.
    //
    //   CARANYA. Proyeksikan matahari ke koordinat layar, lalu dari tiap
    //   piksel telusuri garis lurus MENUJU matahari sambil mencatat berapa
    //   sering garis itu tertutup geometri. Kalau tidak pernah tertutup,
    //   piksel itu duduk di jalur bebas dari matahari -> terang. Kalau
    //   separuhnya tertutup gedung -> setengah terang. Batas antara keduanya
    //   TAJAM karena siluet gedung tajam, dan itulah yang membuatnya terbaca
    //   sebagai berkas, bukan kabut.
    //
    //   Dua-duanya dipertahankan dan dijumlahkan: ray march memberi hamburan
    //   udara yang benar, berkas radial memberi bentuk yang terlihat.
    // ========================================================================
    float radial = 0.0;
    {
        // Matahari diproyeksikan lewat titik sangat jauh searah cahaya.
        vec4 sunClip = u.viewProj * vec4(camPos + L * 1.0e5, 1.0);

        // w <= 0 berarti matahari di BELAKANG kamera. Tanpa uji ini,
        // pembagian menghasilkan koordinat cermin dan berkasnya muncul di
        // sisi layar yang salah — cacat klasik teknik ini.
        if (sunClip.w > 0.0) {
            vec2 sunUV = (sunClip.xy / sunClip.w) * 0.5 + 0.5;

            // Redam saat matahari jauh di luar layar. Tanpa ini, berkasnya
            // muncul mendadak begitu matahari menyentuh tepi layar.
            vec2  d2   = max(abs(sunUV - 0.5) - 0.5, vec2(0.0));
            float edge = 1.0 - smoothstep(0.0, RADIAL_OFFSCREEN_FADE, length(d2));

            if (edge > 0.0) {
                vec2  delta = (uv - sunUV) * (RADIAL_SPAN / float(RADIAL_STEPS));
                vec2  suv   = uv;
                float decay = 1.0;

                // Jitter memakai blue noise yang sama: tanpa ini, 32 langkah
                // tetap menghasilkan pita melingkar di sekitar matahari.
                suv -= delta * jitter;

                for (int i = 0; i < RADIAL_STEPS; ++i) {
                    suv -= delta;
                    vec2 c = clamp(suv, vec2(0.0), vec2(1.0));
                    // >= 1.0 berarti langit: jalur ke matahari bebas di sini.
                    float open = (texture(sceneDepth, c).r >= 1.0) ? 1.0 : 0.0;
                    radial += open * decay;
                    decay  *= RADIAL_DECAY;
                }
                radial *= RADIAL_WEIGHT * edge / float(RADIAL_STEPS);

                // Ikut fungsi fase juga: berkas paling kuat saat menatap
                // matahari, dan itu memang perilaku yang benar.
                radial *= phaseHG(cosT, HG_G) / phaseHG(1.0, HG_G);
            }
        }
    }
    inScatter += radial * RADIAL_STRENGTH;

    imageStore(volOut, coord, vec4(clamp(inScatter, 0.0, 4.0), rayLen, 0.0, 0.0));
}

main.vert:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// main.vert  —  Forward pass.
//   Layout UniformBufferObject SENGAJA dibiarkan identik byte-per-byte dengan
//   struct di main2.cpp, jadi tidak ada perubahan C++ yang dibutuhkan di sini.
//   fragPosLightSpace dihapus: bayangan kini datang dari texture screen-space
//   hasil pipeline compute, bukan dari proyeksi ulang di fragment shader.
// ============================================================================

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4  mvp;
    mat4  lightMVP;       // dipertahankan demi kompatibilitas layout
    vec3  lightPos;
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    float shininess;
    vec3  cameraPos;
    float shadowBias;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormal;

void main() {
    gl_Position  = ubo.mvp * vec4(inPosition, 1.0);
    fragColor    = inColor;
    fragPosWorld = inPosition;   // model matrix = identity di engine ini
    fragNormal   = inNormal;
}

main.frag:

#version 450

// ---- Extension ------------------------------------------------------------
// GL_EXT_control_flow_attributes memberi atribut [[unroll]] dan [[loop]].
// Dipakai pada perulangan yang jumlah putarannya konstan (tap filter, jendela
// 3x3), tempat unroll menghapus seluruh biaya perbandingan dan lompatan per
// putaran. Di rasterizer perangkat lunak, biaya cabang justru sering melebihi
// biaya aritmetikanya sendiri.
//
// : enable, BUKAN : require — kalau driver tidak mendukungnya, atributnya
// diabaikan dan shader tetap dikompilasi. Extension ini murni petunjuk
// optimasi, jadi tidak ada alasan menjadikannya syarat wajib.
#extension GL_EXT_control_flow_attributes : enable
// ============================================================================
// main.frag  —  Forward lighting.
//   Seluruh PCF/Poisson lama DIHAPUS. Bayangan sekarang tinggal satu texture
//   fetch: hasil CSM + PCSS + temporal + bilateral blur yang sudah dihitung
//   di compute pass, sudah ter-filter, sudah stabil secara temporal.
//
//   binding = 1 tetap dipakai — hanya isinya yang berganti dari shadow map
//   depth menjadi shadowFinal (rg16f, full-res, screen-space). Tidak ada
//   perubahan descriptor set layout di C++.
// ============================================================================

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4  mvp;
    mat4  lightMVP;
    vec3  lightPos;
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    float shininess;
    vec3  cameraPos;
    float shadowBias;
} ubo;

// .r  = bayangan matahari (0 terang .. 1 gelap)
// .g  = ambient occlusion (1 terbuka .. 0 tertutup)
// .ba = bent normal, arah rata-rata langit yang masih terbuka (oct, ruang dunia)
layout(binding = 1) uniform sampler2D shadowFinal;

// Light shaft, SETENGAH resolusi. Disampel dengan UV ternormalisasi supaya
// sampler linear yang membesarkannya — tidak perlu pass upsample tersendiri.
// Hamburan volume berfrekuensi sangat rendah, jadi tidak ada detail yang hilang.
layout(binding = 2) uniform sampler2D volumetric;

// Warna matahari untuk berkas cahaya: sedikit hangat, seperti cahaya yang sudah
// melewati atmosfer. Sengaja BUKAN putih — berkas putih murni terbaca sebagai
// kabut, bukan sinar matahari.
const vec3 SUN_SHAFT_COLOR = vec3(1.00, 0.93, 0.78);

// ---------------------------------------------------------------------------
// WARNA MATAHARI (v38)
//
// PERINGATAN JUJUR: ini pencahayaan, bukan mask bayangan. Dipasang justru
// KARENA mask-nya sudah terbukti selesai — penumbra terukur cocok dengan
// kebenaran analitik (17-20 mm vs 21,0 mm prediksi), dan rasio gelap/terang
// sudah setara referensi (0,48 vs 0,45). Yang masih beda cuma HUE-nya.
//
// Sebelum ini matahari vec3(1,0) putih polos, sehingga bayangan dan bagian
// terang cuma berbeda KECERAHAN. Di dunia nyata keduanya berbeda SUMBER:
// bagian terang menerima matahari yang sudah dihangatkan atmosfer, bagian
// bayangan hanya menerima langit yang biru. Perbedaan hue itulah yang membuat
// bayangan terbaca sebagai bayangan, bukan sebagai tambalan gelap.
//
// Angkanya diukur dari screenshot Call of Duty milik user sendiri, bukan
// dikarang. Selisih (lantai terang - lantai bayangan) memisahkan sumbangan
// matahari murni dari ambient:
//
//   mentah        (87,9  65,8  39,2)
//   /hijau        (1,335 1,000 0,595)   luminansi 1,0419
//   dinormalkan   (1,281 0,960 0,571)   luminansi 1,0000
//
// Dibagi luminansinya sendiri supaya KECERAHAN TOTAL GAMBAR TIDAK BERUBAH —
// yang berubah hanya distribusinya antar kanal. Jadi semua angka kalibrasi
// yang sudah dikumpulkan (lantai terang 137, bayangan 71) tetap berlaku.
//
// Kembalikan ke vec3(1.0) kalau tidak suka; tidak ada yang bergantung padanya.
const vec3 SUN_COLOR = vec3(1.281, 0.960, 0.571);

vec3 octDecodeBent(vec2 f) {
    vec3 n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// ---- Ambient hemisfer -------------------------------------------------------
// Ambient sebelumnya PUTIH RATA ke segala arah. Selama begitu, bent normal tidak
// ada gunanya: kalau langit sama di semua arah, mengetahui arahnya tidak
// mengubah apa pun. Jadi keduanya harus datang bersama.
//
// Model paling sederhana yang sudah benar: langit di atas, pantulan tanah di
// bawah. Permukaan menghadap ke atas menerima warna langit, menghadap ke bawah
// menerima pantulan tanah yang lebih hangat dan gelap. Nilainya dipilih agar
// rata-ratanya mendekati 1,0 sehingga kecerahan keseluruhan tidak melompat
// dibanding build sebelumnya.
const vec3 SKY_COLOR    = vec3(0.95, 1.05, 1.30);
const vec3 GROUND_COLOR = vec3(0.75, 0.70, 0.62);

// Batas bawah ambient SETELAH AO dan warna langit dikalikan. Inilah jaminan
// "bagian tergelap tetap menyimpan detail" yang sesungguhnya — lantai pada AO
// saja tidak menjamin apa pun, karena masih ada pengali sesudahnya.
// v46: 0,45 -> 0,72.
//
// Angka lama itulah yang menghasilkan cincin nyaris hitam di kaki bangunan,
// dan cincin itulah yang selama beberapa putaran terbaca sebagai "kebocoran
// cahaya" — bukan karena ada sinar bocor, tapi karena ada cincin yang JAUH
// lebih gelap daripada apa pun di sekitarnya, sehingga tanah terang di
// sebelahnya terlihat seperti menembus.
//
// Dibalik dari frame terakhirmu, bukan dikira-kira. Tanah bayangan penuh
// (AO ~ 1) terukur 71 di layar = 0,0635 linier. Kalau ambFactor jatuh ke
// lantainya:
//
//   lantai 0,45 -> 47,2 di layar     <- terukur di baris kontakmu: 45..55
//   lantai 0,55 -> 52,5
//   lantai 0,65 -> 57,3
//   lantai 0,72 -> 60,6
//   lantai 0,85 -> 65,7
//
// Cocok sampai satu digit. Jadi cincin itu memang AO yang menyentuh lantainya
// sendiri — bekerja persis seperti yang ditulis, cuma lantainya kelewat dalam.
//
// 0,72 dipilih karena di situ titik tergelap (60,6) masih JELAS lebih gelap
// daripada bayangan penuh (71) — jadi kesan AO yang lebih gelap, yang memang
// kamu suka, tetap ada — tapi jarak ke tanah terang di sebelahnya (137)
// mengecil dari 2,9x menjadi 2,3x, dan kontras ekstrem yang bikin tepinya
// terbaca seperti kebocoran itu hilang.
//
// Kalau masih terlalu dalam, naikkan lagi. Kalau AO jadi terasa hambar,
// turunkan. Satu angka, satu arah, tidak ada tebakan lain di baliknya.
const float AMBIENT_FLOOR = 0.72;

void main() {
    // Resolusi diambil dari texture itu sendiri -> tidak perlu menambah field
    // resolusi ke UBO, jadi struct C++ tetap tidak tersentuh.
    vec2  screenUV = gl_FragCoord.xy / vec2(textureSize(shadowFinal, 0));
    vec4  sh       = texture(shadowFinal, screenUV);
    float shadow   = clamp(sh.r, 0.0, 1.0);
    float ao       = clamp(sh.g, 0.0, 1.0);
    vec3  bentN    = octDecodeBent(sh.ba);

    vec3 N = normalize(fragNormal);

    // Directional light, BUKAN point light.
    // Cascade shadow map memakai proyeksi ortografis dari satu arah matahari
    // yang seragam untuk seluruh scene. Kalau di sini tetap memakai
    // normalize(lightPos - fragPosWorld), arah cahaya per-piksel akan menyimpang
    // sampai puluhan derajat di sudut peta, sehingga tepi bayangan tidak berimpit
    // dengan terminator diffuse-nya.
    vec3 L = normalize(ubo.lightPos);

    // PERBAIKAN: view direction dulu di-hardcode ke vec3(0,0,5), sehingga
    // highlight specular tidak ikut bergerak bersama kamera.
    vec3 V = normalize(ubo.cameraPos - fragPosWorld);

    float NdotL = max(dot(N, L), 0.0);

    // Blinn-Phong: halfway vector lebih stabil daripada reflect() pada
    // sudut miring, dan lebih murah.
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), max(ubo.shininess, 1.0));

    // GERBANG NdotL — inilah yang kurang, dan akibatnya terlihat sebagai
    // "cahaya bocor" berupa garis terang setebal satu piksel di puncak
    // bangunan.
    //
    // spec cuma bergantung pada N dan H, tidak pada apakah cahaya benar-benar
    // SAMPAI ke permukaan itu. Muka yang dilihat nyaris dari samping punya N
    // yang bisa saja dekat dengan H walaupun NdotL sudah negatif — jadi ia
    // berkilau padahal matahari ada di belakangnya. Terukur di frame ini:
    // garisnya 172, lebih terang daripada lantai yang kena matahari penuh
    // (115). Tidak ada suku diffuse yang bisa melakukan itu; hanya specular
    // yang lepas kendali bisa.
    //
    // Dikalikan NdotL, bukan sekadar step(): perkaliannya membuat specular
    // meredup MULUS menuju terminator alih-alih terpotong garis tajam — dan
    // di engine tanpa anti-aliasing, potongan tajam persis jenis galat yang
    // akan kembali terbaca sebagai garis satu piksel.
    spec *= NdotL;

    // Ambient occlusion HANYA mengalikan ambient, bukan diffuse maupun specular.
    // AO adalah visibilitas langit; cahaya matahari sudah punya penghalangnya
    // sendiri berupa bayangan CSM. Mengalikannya ke cahaya langsung adalah
    // kesalahan umum yang membuat gambar kotor dan terlalu gelap.
    //
    // Multibounce (Jimenez 2016): AO satu skalar mengasumsikan cahaya yang
    // terhalang HILANG, padahal ia memantul di antara permukaan. Untuk permukaan
    // terang, oklusi yang terasa jauh lebih ringan daripada nilai geometrisnya.
    // Tanpa koreksi ini, sudut ruangan selalu terlihat terlalu pekat.
    vec3 albedo = fragColor;
    vec3 mbA =  2.0404 * albedo - 0.3324;
    vec3 mbB = -4.7951 * albedo + 0.6417;
    vec3 mbC =  2.7552 * albedo + 0.6903;
    vec3 aoMB = max(vec3(ao), ((ao * mbA + mbB) * ao + mbC) * ao);

    // Warna langit diambil menurut BENT NORMAL, bukan normal geometri.
    // Di ruang terbuka keduanya sama. Di dekat dinding, bent normal condong
    // menjauhi dinding — jadi sudut ruangan tidak sekadar lebih gelap, warnanya
    // juga bergeser ke arah langit yang masih terlihat. Itu isyarat kedalaman
    // yang tidak bisa diberikan AO skalar seberapa pun bagusnya.
    vec3 skyLight = mix(GROUND_COLOR, SKY_COLOR, bentN.y * 0.5 + 0.5);

    // ---- Lantai diterapkan pada HASIL AKHIR, bukan pada satu faktor ---------
    //
    // AO_MIN_VISIBILITY = 0,45 dipasang supaya bagian tergelap tetap menyimpan
    // detail. Tapi lantai itu dipasang pada aoMB saja, padahal setelahnya masih
    // ada pengali kedua: skyLight, yang di arah bawah turun sampai ~0,70.
    //
    // Di sudut lantai-dinding keduanya menghantam bersamaan, jadi lantai
    // efektifnya 0,45 x 0,70 = 0,315 — bukan 0,45 seperti yang dimaksudkan.
    // Terukur di layar: garis gelap di kaki dinding cuma 0,32x kecerahan
    // ambient di tempat lain, padahal AO_MIN_VISIBILITY menjanjikan 0,45.
    //
    // Lantai hanya bermakna kalau diterapkan pada apa yang benar-benar sampai
    // ke gambar. Jadi ia dipindah ke sini, sesudah kedua pengali.
    vec3 ambFactor = clamp(aoMB, 0.0, 1.0) * skyLight;
    ambFactor = max(ambFactor, vec3(AMBIENT_FLOOR));

    vec3 ambient = ubo.ambientStrength * ambFactor;
    vec3 diffuse  = ubo.diffuseStrength  * NdotL * SUN_COLOR;
    vec3 specular = ubo.specularStrength * spec  * SUN_COLOR;

    float vis = 1.0 - shadow;

    // Specular ikut dimatikan oleh bayangan; kalau tidak, permukaan di dalam
    // bayangan tetap berkilau dan terlihat "menempel" di atas gambar.
    vec3 result = (ambient + vis * (diffuse + specular)) * fragColor;

    // Light shaft ditambahkan, bukan dikalikan. Ia cahaya yang dihamburkan udara
    // DI ANTARA kamera dan permukaan, jadi ia menumpuk di atas apa pun warna
    // permukaannya — termasuk di atas permukaan yang sepenuhnya dalam bayangan.
    // Itu sebabnya berkasnya tetap terlihat melintasi daerah gelap.
    // ---- Pembesaran SADAR-KEDALAMAN dari setengah resolusi -----------------
    //
    // texture() dengan sampler linear tidak boleh dipakai di sini. Tepat di
    // siluet, satu texel setengah-res menaungi piksel langit sekaligus piksel
    // dinding; interpolasi bilinear lalu mencampur hamburan milik langit — yang
    // raynya penuh 60 m — ke dinding yang raynya pendek. Hasilnya garis terang
    // selebar dua piksel di setiap puncak bangunan.
    //
    // Jadi keempat texel diambil sendiri, dan bobot bilinear-nya dikalikan
    // bobot kesamaan jarak: texel yang panjang raynya jauh berbeda dari jarak
    // fragmen ini ke kamera pasti milik permukaan lain, dan dibuang.
    float shaft;
    {
        float fragDist = length(ubo.cameraPos - fragPosWorld);
        vec2  vs   = vec2(textureSize(volumetric, 0));
        vec2  vuv  = screenUV * vs - 0.5;
        vec2  fr   = fract(vuv);
        ivec2 base = ivec2(floor(vuv));

        // Toleransi sebanding jarak: beda 1 m wajar di 60 m, tidak di 2 m.
        float tol = max(fragDist * 0.10, 0.5);

        float sum = 0.0, wsum = 0.0, nearest = 0.0, bestW = -1.0;
        for (int i = 0; i < 4; ++i) {
            ivec2 off = ivec2(i & 1, i >> 1);
            ivec2 c   = clamp(base + off, ivec2(0), ivec2(vs) - 1);
            vec2  sm  = texelFetch(volumetric, c, 0).rg;

            vec2  bw2 = mix(1.0 - fr, fr, vec2(off));
            float wBil = bw2.x * bw2.y;
            float wDep = exp(-abs(sm.g - fragDist) / tol);
            float w    = wBil * wDep;

            sum  += sm.r * w;
            wsum += w;
            if (wBil > bestW) { bestW = wBil; nearest = sm.r; }
        }
        // Kalau keempatnya ditolak (fragmen betul-betul terisolasi), pakai texel
        // terdekat apa adanya alih-alih menghasilkan nol yang mencolok.
        shaft = (wsum > 1e-4) ? (sum / wsum) : nearest;
    }

    result += shaft * SUN_SHAFT_COLOR;

    outColor = vec4(result, 1.0);
}

sky.vert:

#version 450

// ============================================================================
// sky.vert — segitiga layar penuh, tanpa vertex buffer.
//
//   gl_Position.z = 1.0 disengaja: pipeline langit memakai depth test
//   LESS_OR_EQUAL dengan depth write MATI, dan depth buffer di-clear ke 1.0.
//   Jadi fragmen langit hanya lolos di piksel yang TIDAK ditulis geometri —
//   nol overdraw, dan langit otomatis terhalang bangunan tanpa satu pun uji
//   tambahan di fragment shader.
//
//   vFarH dikirim sebagai vec4 HOMOGEN, bukan arah yang sudah dibagi w.
//   Alasannya presisi, bukan gaya: pemetaan dari NDC ke ruang dunia bersifat
//   proyektif, jadi menginterpolasi hasil pembagiannya salah. Menginterpolasi
//   vec4-nya lalu membagi PER FRAGMEN justru persis benar — itu mekanisme yang
//   sama dengan interpolasi perspektif biasa.
// ============================================================================

layout(binding = 0) uniform UniformBufferObject {
    mat4  mvp;
    mat4  lightMVP;
    vec3  lightPos;
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    float shininess;
    vec3  cameraPos;
    float shadowBias;
} ubo;

layout(location = 0) out vec4 vFarH;

void main() {
    // (0,0) (2,0) (0,2) -> segitiga yang menutupi seluruh layar
    vec2 uv  = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vec2 ndc = uv * 2.0 - 1.0;

    gl_Position = vec4(ndc, 1.0, 1.0);

    // ubo.mvp adalah viewProj BER-JITTER (model = identitas). Memakai matriks
    // yang sama dengan geometri itu wajib: kalau langit tidak ikut ter-jitter,
    // TAA akan melihat langit yang diam di atas geometri yang bergoyang.
    vFarH = inverse(ubo.mvp) * vec4(ndc, 1.0, 1.0);
}

sky.frag:

#version 450

#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// sky.frag — langit analitik Preetham + berkas cahaya volumetrik
//
//   MENGAPA PASS INI ADA.
//
//   volumetric.comp menghitung hamburan untuk SETIAP piksel, termasuk piksel
//   langit. Piksel langit justru yang nilainya paling besar: raynya tidak
//   pernah berhenti di permukaan, jadi rayLen = VOL_MAX_DIST penuh, dan
//   inScatter berbanding lurus dengan rayLen. Terukur dari konstanta yang
//   sedang berlaku (density 0,012, maxDist 60 m, HG g=0,6):
//
//       piksel langit menghadap matahari  : 0,5730
//       piksel langit 45 derajat          : 0,1002
//
//   Nilai itu tidak pernah dibaca siapa pun. main.frag satu-satunya konsumen
//   buffer volumetric, dan ia hanya berjalan pada fragmen GEOMETRI. Piksel
//   langit tidak punya fragmen sama sekali — ia tetap berisi clear color
//   (0,10 0,10 0,15) dari awal sampai akhir.
//
//   Artinya berkas cahaya dihitung penuh tiap frame lalu dibuang persis di
//   tempat ia paling terlihat: melawan langit. Itu bukan kekurangan setelan,
//   itu konsumen yang tidak pernah ada. Pass ini konsumennya.
//
//   MENGAPA LANGITNYA MODEL, BUKAN GRADIEN.
//
//   Berkas cahaya cuma terbaca kalau latarnya benar. Gradien buatan tangan
//   memberi dua masalah: warnanya tidak bergantung pada posisi matahari, jadi
//   berkas yang memancar dari satu titik akan terlihat menempel di latar yang
//   tidak tahu-menahu soal titik itu; dan kecerahannya tidak punya satuan,
//   jadi tidak bisa diserasikan dengan ambient.
//
//   Preetham (1999) adalah model langit analitik: satu fungsi tertutup, tanpa
//   ray march, yang memberi luminansi DAN kromatisitas untuk tiap arah pandang
//   sebagai fungsi sudut matahari dan kekeruhan udara. Ia menghasilkan sendiri
//   tiga hal yang membuat langit terbaca nyata — zenit biru pekat, horizon
//   memucat karena hamburan Mie, dan aureole terang mengelilingi matahari.
//   Aureole itu penting di sini: ia yang membuat pangkal berkas cahaya
//   menyatu dengan langit alih-alih terpotong.
//
//   Matahari di engine ini elevasinya TETAP (y=12, jari-jari horizontal 20 ->
//   31 derajat), hanya azimutnya berputar. Jadi luminansi zenit konstan
//   sepanjang permainan dan tidak ada kejutan terang-gelap saat matahari
//   berputar.
// ============================================================================

layout(location = 0) in  vec4 vFarH;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4  mvp;
    mat4  lightMVP;
    vec3  lightPos;
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    float shininess;
    vec3  cameraPos;
    float shadowBias;
} ubo;

// Dipakai HANYA untuk textureSize: ia satu-satunya texture beresolusi penuh
// yang sudah ada di descriptor set ini, dan pembesaran bilateral di bawah
// butuh resolusi layar yang tepat. Mengambilnya dari sini menghindari
// menambah field resolusi ke UBO.
layout(binding = 1) uniform sampler2D shadowFinal;
layout(binding = 2) uniform sampler2D volumetric;

layout(push_constant) uniform PC {
    // x = turbidity (kekeruhan udara)
    // y = luminansi zenit dalam satuan engine
    // z = kecerahan cakram matahari
    // w = pengali berkas cahaya di langit
    vec4 p;
} pc;

// HARUS sama dengan main.frag. Kalau salah satu diubah, yang lain ikut —
// kalau tidak, berkas yang sama akan berganti warna saat menyeberangi siluet
// bangunan, dan itu jenis cacat yang sangat mencolok.
const vec3  SUN_SHAFT_COLOR    = vec3(1.00, 0.93, 0.78);

// Jari-jari sudut matahari, radian. tan(0,515 derajat) = 0,009 — nilai yang
// sama dengan Cfg::SUN_ANGULAR_TAN yang dipakai PCSS untuk lebar penumbra.
// Keduanya harus sepakat: cakram matahari yang terlihat di langit dan
// kelembutan tepi bayangan di lantai berasal dari sudut yang sama.
const float SUN_ANGULAR_RADIUS = 0.004596;   // = atan(Cfg::SUN_ANGULAR_TAN)
const float LIMB_DARKENING     = 0.6;      // piringan matahari meredup di tepi

// ---------------------------------------------------------------------------
// Fungsi Perez. Dua suku yang terpisah tugasnya:
//   (1 + A exp(B/cos(theta)))     gradien zenit -> horizon
//   (1 + C exp(D gamma) + E cos^2 gamma)   aureole di sekitar matahari
// cos(theta) dijaga di 0,01: tepat di horizon pembaginya menuju nol.
// ---------------------------------------------------------------------------
float perez(float A, float B, float C, float D, float E, float cosTheta, float gamma) {
    float ct = max(cosTheta, 0.01);
    float cg = cos(gamma);
    return (1.0 + A * exp(B / ct)) * (1.0 + C * exp(D * gamma) + E * cg * cg);
}

vec3 preethamSky(vec3 dir, vec3 sunDir, float T) {
    float cosTheta = max(dir.y, 0.0);                       // 1 di zenit
    float thetaS   = acos(clamp(sunDir.y, 0.0, 1.0));       // sudut zenit matahari
    float gamma    = acos(clamp(dot(dir, sunDir), -1.0, 1.0));

    // Koefisien distribusi, linier terhadap turbidity (Preetham tabel 1).
    float AY =  0.1787 * T - 1.4630;
    float BY = -0.3554 * T + 0.4275;
    float CY = -0.0227 * T + 5.3251;
    float DY =  0.1206 * T - 2.5771;
    float EY = -0.0670 * T + 0.3703;

    float Ax = -0.0193 * T - 0.2592;
    float Bx = -0.0665 * T + 0.0008;
    float Cx = -0.0004 * T + 0.2125;
    float Dx = -0.0641 * T - 0.8989;
    float Ex = -0.0033 * T + 0.0452;

    float Ay = -0.0167 * T - 0.2608;
    float By = -0.0950 * T + 0.0092;
    float Cy = -0.0079 * T + 0.2102;
    float Dy = -0.0441 * T - 1.6537;
    float Ey = -0.0109 * T + 0.0529;

    // Nilai di zenit: jangkar absolut yang diskalakan fungsi Perez.
    float t  = thetaS;
    float t2 = t * t;
    float t3 = t2 * t;
    float T2 = T * T;

    float xz = ( 0.00166*t3 - 0.00375*t2 + 0.00209*t          ) * T2
             + (-0.02903*t3 + 0.06377*t2 - 0.03202*t + 0.00394) * T
             + ( 0.11693*t3 - 0.21196*t2 + 0.06052*t + 0.25886);

    float yz = ( 0.00275*t3 - 0.00610*t2 + 0.00317*t          ) * T2
             + (-0.04214*t3 + 0.08970*t2 - 0.04153*t + 0.00516) * T
             + ( 0.15346*t3 - 0.26756*t2 + 0.06670*t + 0.26688);

    float chi = (4.0/9.0 - T/120.0) * (3.14159265 - 2.0 * t);
    float Yz  = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;

    float Y = Yz * perez(AY,BY,CY,DY,EY, cosTheta, gamma)
                 / perez(AY,BY,CY,DY,EY, 1.0,      thetaS);
    float x = xz * perez(Ax,Bx,Cx,Dx,Ex, cosTheta, gamma)
                 / perez(Ax,Bx,Cx,Dx,Ex, 1.0,      thetaS);
    float y = yz * perez(Ay,By,Cy,Dy,Ey, cosTheta, gamma)
                 / perez(Ay,By,Cy,Dy,Ey, 1.0,      thetaS);

    // xyY -> XYZ -> sRGB linier.
    y = max(y, 1e-4);
    float X = (x / y) * Y;
    float Z = ((1.0 - x - y) / y) * Y;

    vec3 rgb = vec3( 3.2406*X - 1.5372*Y - 0.4986*Z,
                    -0.9689*X + 1.8758*Y + 0.0415*Z,
                     0.0557*X - 0.2040*Y + 1.0570*Z);

    // Preetham memberi luminansi absolut dalam kcd/m2 — satuan yang tidak
    // berarti apa-apa bagi renderer ini. Dibagi luminansi zenitnya sendiri,
    // bentuk dan warna langit tetap utuh sementara skalanya jadi satu angka
    // yang bisa diserasikan dengan sisa gambar.
    rgb *= pc.p.y / max(Yz, 1e-4);

    // Kromatisitas jenuh bisa jatuh sedikit di luar gamut sRGB dan
    // menghasilkan kanal negatif. Negatif akan menembus tonemap dan muncul
    // sebagai bercak, jadi dipotong di sini.
    return max(rgb, vec3(0.0));
}

void main() {
    // Bagi w PER FRAGMEN — lihat catatan di sky.vert.
    vec3 dir = normalize(vFarH.xyz / vFarH.w - ubo.cameraPos);
    vec3 S   = normalize(ubo.lightPos);

    float T   = clamp(pc.p.x, 1.7, 10.0);   // di bawah 1,7 modelnya divergen
    vec3  col = preethamSky(dir, S, T);

    // ---- Di bawah horizon --------------------------------------------------
    // Preetham hanya terdefinisi untuk belahan atas. cos(theta) sudah dijepit
    // di nol, jadi arah ke bawah menerima nilai horizon; yang kurang tinggal
    // peredupan supaya ia terbaca sebagai tanah berkabut, bukan langit
    // terbalik. Transisinya dibuat lembut selebar 0,06 satuan: engine ini
    // tidak punya MSAA, jadi setiap potongan tajam akan kembali sebagai garis
    // satu piksel.
    col *= mix(1.0, 0.30, smoothstep(0.0, -0.06, dir.y));

    // ---- Cakram matahari ---------------------------------------------------
    // Aureole-nya sudah dihasilkan suku Perez; yang ditambahkan di sini cuma
    // piringannya sendiri. Limb darkening dipakai karena murah dan benar:
    // tepi piringan matahari memang lebih redup daripada pusatnya, sebab garis
    // pandang di tepi menembus lapisan fotosfer yang lebih dingin.
    float gamma = acos(clamp(dot(dir, S), -1.0, 1.0));
    float r     = gamma / SUN_ANGULAR_RADIUS;
    if (r < 1.2 && S.y > 0.0) {
        float rc   = min(r, 1.0);
        float mu   = sqrt(max(1.0 - rc * rc, 0.0));
        float limb = 1.0 - LIMB_DARKENING * (1.0 - mu);
        // Tepi dilembutkan menyeberangi ~0,3 jari-jari. Tanpa ini piringan
        // matahari jadi cakram bergerigi — satu-satunya benda di layar dengan
        // tepi kurva keras, jadi geriginya justru paling terlihat.
        float edge = 1.0 - smoothstep(0.85, 1.15, r);
        col += vec3(1.00, 0.96, 0.90) * (pc.p.z * limb * edge);
    }

    // ---- Berkas cahaya, pembesaran SADAR-JARAK -----------------------------
    //
    // Cerminan dari blok yang sama di main.frag, dengan satu perbedaan yang
    // menentukan: di sana acuannya jarak fragmen ke kamera, di sini fragmennya
    // langit dan tidak punya jarak.
    //
    // Acuan yang dipakai: panjang ray TERPANJANG di antara empat texel.
    // Piksel langit selalu memegang rayLen maksimum, jadi di tepi siluet texel
    // milik bangunan (rayLen pendek) otomatis tertolak. Tanpa penolakan ini,
    // bilinear polos akan menyeret hamburan langit ke dalam dinding — cacat
    // yang persis sama dengan garis terang dua piksel di puncak bangunan.
    vec2  fullRes  = vec2(textureSize(shadowFinal, 0));
    vec2  screenUV = gl_FragCoord.xy / fullRes;
    vec2  vs   = vec2(textureSize(volumetric, 0));
    vec2  vuv  = screenUV * vs - 0.5;
    vec2  fr   = fract(vuv);
    ivec2 base = ivec2(floor(vuv));

    float sVal[4], gVal[4], wBil[4];
    float maxLen = 0.0;
    [[unroll]] for (int i = 0; i < 4; ++i) {
        ivec2 off = ivec2(i & 1, i >> 1);
        ivec2 c   = clamp(base + off, ivec2(0), ivec2(vs) - 1);
        vec2  sm  = texelFetch(volumetric, c, 0).rg;
        sVal[i]   = sm.r;
        gVal[i]   = sm.g;
        vec2 bw   = mix(1.0 - fr, fr, vec2(off));
        wBil[i]   = bw.x * bw.y;
        maxLen    = max(maxLen, sm.g);
    }

    float tol = max(maxLen * 0.10, 0.5);
    float sum = 0.0, wsum = 0.0, nearest = 0.0, bestW = -1.0;
    [[unroll]] for (int i = 0; i < 4; ++i) {
        float w = wBil[i] * exp(-(maxLen - gVal[i]) / tol);
        sum  += sVal[i] * w;
        wsum += w;
        if (wBil[i] > bestW) { bestW = wBil[i]; nearest = sVal[i]; }
    }
    float shaft = (wsum > 1e-4) ? (sum / wsum) : nearest;

    // Ditambahkan, bukan dikalikan: hamburan adalah cahaya yang datang DARI
    // udara di antara kamera dan latar, jadi ia menumpuk di atas langit.
    col += shaft * SUN_SHAFT_COLOR * pc.p.w;

    outColor = vec4(col, 1.0);
}

taa_resolve.comp:

#version 450

// ---- Extension ------------------------------------------------------------
#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// taa_resolve.comp — Temporal Anti-Aliasing
//
//   Setiap frame, matriks proyeksi digeser kurang dari satu piksel mengikuti
//   deret Halton. Jadi frame 1 menyampel titik yang sedikit berbeda dari frame
//   2, dan seterusnya. Menumpuk deretan itu memberi informasi SUB-PIKSEL yang
//   tidak pernah dimiliki satu frame pun — dan itulah anti-aliasing-nya.
//
//   Yang membuat TAA sulit bukan penumpukannya, melainkan tahu KAPAN HARUS
//   BERHENTI menumpuk. Kalau piksel yang sama menampilkan benda berbeda dari
//   frame sebelumnya, history-nya racun. Tiga penjaga di bawah yang mengurusnya.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D currColor;   // hdr frame ini (ber-jitter)
layout(binding = 1) uniform sampler2D historyIn;   // hasil TAA frame lalu
layout(binding = 2) uniform sampler2D velocityTex; // rg16f dari pre-pass
layout(binding = 3) uniform sampler2D sceneDepth;
layout(binding = 4, rgba16f) uniform writeonly image2D historyOut;

layout(push_constant) uniform PC {
    // x = blend maksimum (porsi history saat diam)
    // y = ketajaman unsharp
    // z = 1 = TAA aktif, 0 = lewat saja
    // w = deltaTime (detik)
    vec4 p;
} pc;

// Ruang YCoCg, bukan RGB.
//
// Penjepitan neighborhood bekerja jauh lebih baik di ruang yang memisahkan
// terang dari warna: kotak batas di RGB harus melebar mengikuti kanal yang
// paling bervariasi, sehingga ia terlalu longgar untuk dua kanal lainnya dan
// meloloskan ghosting. Di YCoCg, luminance punya batasnya sendiri — dan mata
// paling peka justru pada luminance.
vec3 rgb2ycocg(vec3 c) {
    return vec3( 0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
                 0.5  * c.r            - 0.5  * c.b,
                -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
vec3 ycocg2rgb(vec3 c) {
    float t = c.x - c.z;
    return vec3(t + c.y, c.x + c.z, t - c.y);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim   = textureSize(currColor, 0);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    vec3 curr = texelFetch(currColor, coord, 0).rgb;

    if (pc.p.z < 0.5) {                       // TAA dimatikan
        imageStore(historyOut, coord, vec4(curr, 1.0));
        return;
    }

    vec2 texel = 1.0 / vec2(dim);
    vec2 uv    = (vec2(coord) + 0.5) * texel;

    // ---- Statistik tetangga 3x3, di YCoCg --------------------------------
    // Sekaligus mencari piksel dengan kedalaman TERDEKAT ke kamera. Itu dipakai
    // untuk mengambil motion vector: di tepi objek yang bergerak, motion vector
    // milik piksel terdepan yang benar — kalau memakai motion vector piksel
    // tengah, seluruh tepi objek meninggalkan jejak.
    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    float bestDepth = 1.0;
    ivec2 bestCoord = coord;

    [[unroll]] for (int dy = -1; dy <= 1; ++dy)
    [[unroll]] for (int dx = -1; dx <= 1; ++dx) {
        ivec2 c = clamp(coord + ivec2(dx, dy), ivec2(0), dim - 1);
        vec3  s = rgb2ycocg(texelFetch(currColor, c, 0).rgb);
        m1 += s;
        m2 += s * s;

        float d = texelFetch(sceneDepth, c, 0).r;
        if (d < bestDepth) { bestDepth = d; bestCoord = c; }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    vec3 sigma = sqrt(max(m2 - m1 * m1, vec3(0.0)));

    // Gamma variance clipping. 1.0 ketat (buang ghosting, sisakan sedikit
    // kedip), 1.5 longgar (lebih mantap, risiko jejak). 1.25 titik tengah yang
    // dipakai banyak engine.
    const float GAMMA = 1.25;
    vec3 boxMin = m1 - GAMMA * sigma;
    vec3 boxMax = m1 + GAMMA * sigma;

    // ---- Reproyeksi -------------------------------------------------------
    // Motion vector sudah BEBAS JITTER: matriks frame sebelumnya dibangun ulang
    // dengan jitter frame INI, sehingga suku jitter-nya saling menghapus di
    // pre-pass. Tanpa itu, velocity mengandung goyangan sub-piksel jitter dan
    // TAA akan mengejar bayangannya sendiri.
    vec2 vel    = texelFetch(velocityTex, bestCoord, 0).rg;
    vec2 prevUV = uv - vel;

    if (any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0)))) {
        imageStore(historyOut, coord, vec4(curr, 1.0));   // keluar layar
        return;
    }

    // Sampel history dengan filter Catmull-Rom 5 tap.
    //
    // Bilinear biasa memburamkan history sedikit SETIAP frame, dan karena
    // history diumpankan balik ke dirinya sendiri, keburaman itu menumpuk
    // sampai gambar terlihat berlendir. Catmull-Rom punya lobus negatif yang
    // mempertahankan ketajaman melintasi ratusan frame.
    vec2 pos = prevUV * vec2(dim);
    vec2 tc  = floor(pos - 0.5) + 0.5;
    vec2 f   = pos - tc;
    vec2 f2 = f * f, f3 = f2 * f;

    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;
    vec2 w12 = w1 + w2;
    vec2 t12 = w2 / max(w12, vec2(1e-5));

    vec2 uv0  = (tc - 1.0)      * texel;
    vec2 uv12 = (tc + t12)      * texel;
    vec2 uv3  = (tc + 2.0)      * texel;

    vec3 hist =
        texture(historyIn, vec2(uv12.x, uv0.y )).rgb * (w12.x * w0.y ) +
        texture(historyIn, vec2(uv0.x,  uv12.y)).rgb * (w0.x  * w12.y) +
        texture(historyIn, vec2(uv12.x, uv12.y)).rgb * (w12.x * w12.y) +
        texture(historyIn, vec2(uv3.x,  uv12.y)).rgb * (w3.x  * w12.y) +
        texture(historyIn, vec2(uv12.x, uv3.y )).rgb * (w12.x * w3.y );

    float wsum = (w12.x*w0.y) + (w0.x*w12.y) + (w12.x*w12.y)
               + (w3.x*w12.y) + (w12.x*w3.y);
    hist /= max(wsum, 1e-5);
    hist = max(hist, vec3(0.0));

    // ---- Penjepitan ke kotak tetangga -------------------------------------
    vec3 histY = rgb2ycocg(hist);
    vec3 clipped = clamp(histY, boxMin, boxMax);

    // Seberapa jauh history harus dipaksa masuk kotak = ukuran seberapa tidak
    // cocok ia. Makin dipaksa, makin sedikit ia dipercaya.
    float reject = length((histY - clipped) / max(sigma, vec3(1e-3)));
    float trust  = exp(-reject * 0.5);

    // Kecepatan piksel per DETIK, bukan per frame. Sama seperti temporal
    // bayangan: ambang per frame membuat perilaku TAA berubah-ubah mengikuti
    // FPS, dan pada FPS rendah ia menolak history terus-menerus.
    float speedPS = length(vel * vec2(dim)) / max(pc.p.w, 1e-4);
    float speedFade = 1.0 - smoothstep(60.0, 1200.0, speedPS);

    float blend = pc.p.x * trust * speedFade;

    // ---- BATAS ATAS BERBASIS WAKTU -----------------------------------------
    //
    // pc.p.x adalah porsi history PER FRAME. Itu berbasis jumlah frame, bukan
    // detik — cacat yang sama dengan yang baru diperbaiki di shadow_temporal.
    //
    //   60  FPS: 0,92 per frame -> riwayat memudar ~0,5 detik   (benar)
    //   0,5 FPS: 0,92 per frame -> riwayat memudar ~55 detik    (ghosting)
    //
    // Satu frame di rasterizer perangkat lunak bisa dua detik. Menahan 92%
    // riwayat selama itu berarti gambar lama menempel puluhan detik.
    //
    // exp(-dt/TAU) mengubah "porsi per frame" jadi "waktu memudar tetap":
    // di 60 FPS ia longgar (0,94) dan pc.p.x yang menentukan; di 0,5 FPS ia
    // runtuh ke 0,0003 sehingga TAA mematikan dirinya sendiri. Tanpa sakelar,
    // tanpa konstanta yang harus disetel ulang saat pindah perangkat.
    const float TAA_TAU_SECONDS = 0.25;
    float blendTime = exp(-max(pc.p.w, 1e-4) / TAA_TAU_SECONDS);
    blend = min(blend, blendTime);

    vec3 res = ycocg2rgb(mix(rgb2ycocg(curr), clipped, clamp(blend, 0.0, 0.98)));

    // ---- TIDAK ADA penajaman di sini ---------------------------------------
    //
    // Dulu ada unsharp di titik ini, dan itulah sumber bercak gelap padat di
    // dinding.
    //
    // Sebabnya: hasilnya ditulis ke historyOut, dan history diumpankan balik ke
    // dirinya sendiri frame berikutnya. Jadi unsharp diterapkan LAGI di atas
    // hasil yang sudah di-unsharp, berulang tiap frame. Selisih kecil apa pun
    // di sekitar takik geometri diperkuat secara eksponensial:
    //
    //     frame 1: 0,91   frame 3: 0,85   frame 8: 0,38   frame 12: 0,00
    //
    // Konvergen ke nol — jauh di bawah nilai apa pun yang bisa dihasilkan
    // bayangan, yang batas bawahnya ambient. Itu sebabnya bercaknya 55 padahal
    // bayangan penuh saja 114, dan kenapa variance clipping tidak menangkapnya:
    // clipping membandingkan history dengan tetangganya di frame INI, sementara
    // yang meledak justru nilai yang baru saja ditulis sebagai history.
    //
    // Penajaman sekarang dikerjakan di composite.frag, di luar loop — ia membaca
    // history tapi hasilnya tidak pernah ditulis kembali ke sana.
    imageStore(historyOut, coord, vec4(res, 1.0));
}

composite.vert:

#version 450

// ---- Extension ------------------------------------------------------------
#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// composite.vert — segitiga layar penuh TANPA vertex buffer
//
//   Tiga verteks dibangkitkan dari gl_VertexIndex saja. Bukan quad dua segitiga:
//   satu segitiga besar yang menutupi layar lebih murah, karena tidak ada
//   diagonal di tengah layar tempat dua segitiga bertemu — di sana GPU
//   menjalankan kuad piksel ganda.
//
//   Titik: (-1,-1), (3,-1), (-1,3). UV: (0,0), (2,0), (0,2).
// ============================================================================

layout(location = 0) out vec2 vUV;

void main() {
    vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}

composite.frag:

#version 450

// ---- Extension ------------------------------------------------------------
#extension GL_EXT_control_flow_attributes : enable

// ============================================================================
// composite.frag — tonemap HDR -> layar
//
//   Sampai sekarang main pass menggambar LANGSUNG ke swapchain, yang formatnya
//   8 bit per kanal dan terpotong keras di 1.0. Artinya setiap nilai di atas
//   satu — sorotan matahari, berkas volumetric yang menumpuk — hilang begitu
//   saja sebelum sempat dipakai.
//
//   Sekarang main pass menggambar ke target rgba16f, dan pass ini yang
//   memampatkan rentang lebar itu ke layar. Perubahan strukturalnya penting
//   bukan cuma demi tonemap: buffer warna offscreen adalah SYARAT untuk TAA,
//   bloom, dan seluruh post-processing berikutnya, karena semuanya perlu
//   membaca warna frame sebelum ia sampai ke layar.
// ============================================================================

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrColor;

layout(push_constant) uniform PC {
    vec4 params;   // x=exposure  y=mode(0 lewat,1 ACES)  z=gammaManual(0/1)  w=sharpen
    // v44 FXAA: x = subpix (<0 berarti MATI), y = ambang tepi relatif,
    //           z = ambang tepi mutlak, w = cadangan
    vec4 fxaa;
} pc;

// Pendekatan kurva ACES filmic (Narkowicz 2015).
//
// Dipilih bukan karena paling akurat, tapi karena bentuk kurvanya benar:
// bahu yang melandai di ujung terang, sehingga nilai jauh di atas 1.0 tetap
// menghasilkan beda yang terlihat alih-alih rata putih. Reinhard sederhana
// membuat gambar terasa kusam; pemotongan keras membuang detail sorotan.
vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ============================================================================
// FXAA 3.11, preset kualitas 12 (Timothy Lottes, NVIDIA)
//
//   MENGAPA FXAA DAN BUKAN TAA.
//
//   TAA sudah ada di engine ini dan sudah nyala, tapi ia hidup dari frame rate:
//   m_taaHealth diturunkan dari deltaTime, dan di llvmpipe 0,5 FPS nilainya
//   praktis nol, jadi jitter maupun akumulasi riwayatnya tidak pernah bekerja.
//   Terukur akibatnya: transisi siluet 0 PIKSEL — 227 227 227 lalu langsung
//   114 114 114, tanpa satu pun nilai antara.
//
//   FXAA tidak punya riwayat. Ia bekerja penuh dalam satu frame, jadi ia satu-
//   satunya anti-aliasing yang tetap berfungsi di frame rate berapa pun.
//
//   MENGAPA LUMA-NYA sqrt().
//
//   FXAA mengasumsikan masukan dalam ruang gamma: ambang kontrasnya (0,125 dan
//   0,0833) dikalibrasi terhadap persepsi, bukan terhadap energi linier.
//   Swapchain engine ini format sRGB, jadi shader menulis nilai LINIER dan
//   perangkat keras yang mengonversi. Memakai luma linier akan membuat ambang
//   terlalu longgar di daerah gelap dan terlalu ketat di daerah terang.
//   sqrt() adalah gamma 2.0 — pendekatan sRGB yang cukup dekat dan jauh lebih
//   murah daripada pow(1/2.2) yang dipanggil puluhan kali per piksel.
//
//   BIAYA.
//
//   Mayoritas layar keluar lebih awal di uji kontras: piksel datar membayar 5
//   fetch dan berhenti. Hanya piksel tepi yang membayar penelusuran 12 langkah
//   dua arah. Itu sebabnya ambang MUTLAK (FXAA_EDGE_THRESHOLD_MIN) penting —
//   ia yang menjaga langit dan lantai polos tidak ikut membayar.
// ============================================================================

// Tabel langkah preset 12. Langkah membesar di ujung supaya tepi yang sangat
// panjang tetap terjangkau tanpa menambah iterasi di tepi pendek.
const float FXAA_STEP[12] = float[12](
    1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);

vec3 tonemapAt(vec2 uv) {
    vec3 c = texture(hdrColor, uv).rgb * pc.params.x;
    c = (pc.params.y > 0.5) ? acesFilm(c) : clamp(c, 0.0, 1.0);
    if (pc.params.z > 0.5) c = pow(c, vec3(1.0 / 2.2));
    return c;
}

float lumaAt(vec2 uv) {
    // sqrt(luminansi) — lihat catatan gamma di atas.
    return sqrt(dot(tonemapAt(uv), vec3(0.2126, 0.7152, 0.0722)));
}

vec3 fxaa(vec2 posM, vec3 centerColor, vec2 rcpFrame) {
    float lumaM = sqrt(dot(centerColor, vec3(0.2126, 0.7152, 0.0722)));

    float lumaN = lumaAt(posM + vec2(     0.0, -rcpFrame.y));
    float lumaS = lumaAt(posM + vec2(     0.0,  rcpFrame.y));
    float lumaE = lumaAt(posM + vec2( rcpFrame.x,      0.0));
    float lumaW = lumaAt(posM + vec2(-rcpFrame.x,      0.0));

    float rangeMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    float rangeMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float range    = rangeMax - rangeMin;

    // ---- Keluar awal: piksel datar --------------------------------------
    // Ini yang membuat FXAA murah. Dua ambang, dan yang lebih besar menang:
    // yang relatif menjaga tepi berkontras rendah di daerah terang tetap
    // tertangkap, yang mutlak menjaga derau di daerah gelap tidak dikira tepi.
    if (range < max(pc.fxaa.z, rangeMax * pc.fxaa.y)) return centerColor;

    float lumaNW = lumaAt(posM + vec2(-rcpFrame.x, -rcpFrame.y));
    float lumaNE = lumaAt(posM + vec2( rcpFrame.x, -rcpFrame.y));
    float lumaSW = lumaAt(posM + vec2(-rcpFrame.x,  rcpFrame.y));
    float lumaSE = lumaAt(posM + vec2( rcpFrame.x,  rcpFrame.y));

    // ---- Arah tepi: mendatar atau tegak? ---------------------------------
    // Turunan kedua di tiga baris/kolom, baris tengah diberi bobot ganda.
    // Membandingkan dua jumlah ini jauh lebih tahan derau daripada sekadar
    // melihat selisih satu pasang piksel.
    float lumaNS   = lumaN  + lumaS;
    float lumaWE   = lumaW  + lumaE;
    float lumaNWNE = lumaNW + lumaNE;
    float lumaSWSE = lumaSW + lumaSE;
    float lumaNWSW = lumaNW + lumaSW;
    float lumaNESE = lumaNE + lumaSE;

    float edgeHorz1 = (-2.0 * lumaM) + lumaNS;
    float edgeHorz2 = (-2.0 * lumaE) + lumaNESE;
    float edgeHorz3 = (-2.0 * lumaW) + lumaNWSW;
    float edgeVert1 = (-2.0 * lumaM) + lumaWE;
    float edgeVert2 = (-2.0 * lumaN) + lumaNWNE;
    float edgeVert3 = (-2.0 * lumaS) + lumaSWSE;

    float edgeHorz = abs(edgeHorz3) + (abs(edgeHorz1) * 2.0) + abs(edgeHorz2);
    float edgeVert = abs(edgeVert3) + (abs(edgeVert1) * 2.0) + abs(edgeVert2);
    bool  horzSpan = edgeHorz >= edgeVert;

    // ---- Koreksi subpiksel ------------------------------------------------
    // Rata-rata 3x3 dibandingkan piksel tengah: kalau tengahnya menyimpang
    // jauh dari tetangganya, ia fitur sub-piksel (kabel tipis, sliver seperti
    // rusuk menara di scene ini) yang tidak punya tepi panjang untuk
    // ditelusuri. Kurva pangkat empat membuat koreksinya hanya menyala saat
    // penyimpangannya benar-benar besar.
    float subpixA = (lumaNS + lumaWE) * 2.0 + lumaNWNE + lumaSWSE;
    float subpixB = (subpixA * (1.0 / 12.0)) - lumaM;
    float subpixC = clamp(abs(subpixB) / range, 0.0, 1.0);
    float subpixD = (-2.0 * subpixC) + 3.0;
    float subpixE = subpixC * subpixC;
    float subpixF = subpixD * subpixE;
    float subpixH = (subpixF * subpixF) * pc.fxaa.x;

    // ---- Sisi mana dari tepi yang lebih curam? ----------------------------
    float lumaP = horzSpan ? lumaS : lumaE;   // arah "positif" tegak lurus tepi
    float lumaNn = horzSpan ? lumaN : lumaW;
    float lengthSign = horzSpan ? rcpFrame.y : rcpFrame.x;

    float gradientP = lumaP  - lumaM;
    float gradientN = lumaNn - lumaM;
    bool  pairN     = abs(gradientN) >= abs(gradientP);
    float gradient  = max(abs(gradientN), abs(gradientP)) * 0.25;
    if (pairN) lengthSign = -lengthSign;

    // Luma acuan = rata-rata piksel tengah dan tetangga di sisi yang dipilih.
    // Penelusuran berhenti saat luma menyimpang dari acuan ini.
    float lumaAvg = 0.5 * ((pairN ? lumaNn : lumaP) + lumaM);

    // Titik awal penelusuran: setengah texel ke arah tepi, supaya sampelnya
    // jatuh DI ATAS garis tepi, bukan di salah satu sisinya.
    vec2 posB = posM;
    if (horzSpan) posB.y += lengthSign * 0.5;
    else          posB.x += lengthSign * 0.5;

    vec2 offNP = horzSpan ? vec2(rcpFrame.x, 0.0) : vec2(0.0, rcpFrame.y);

    // ---- Penelusuran ujung tepi, dua arah ---------------------------------
    vec2  posN = posB - offNP * FXAA_STEP[0];
    vec2  posP = posB + offNP * FXAA_STEP[0];
    float lumaEndN = lumaAt(posN) - lumaAvg;
    float lumaEndP = lumaAt(posP) - lumaAvg;
    bool  doneN = abs(lumaEndN) >= gradient;
    bool  doneP = abs(lumaEndP) >= gradient;
    float distN = horzSpan ? (posM.x - posN.x) : (posM.y - posN.y);
    float distP = horzSpan ? (posP.x - posM.x) : (posP.y - posM.y);

    [[loop]] for (int i = 1; i < 12; ++i) {
        if (doneN && doneP) break;
        if (!doneN) {
            posN -= offNP * FXAA_STEP[i];
            lumaEndN = lumaAt(posN) - lumaAvg;
            doneN = abs(lumaEndN) >= gradient;
            distN = horzSpan ? (posM.x - posN.x) : (posM.y - posN.y);
        }
        if (!doneP) {
            posP += offNP * FXAA_STEP[i];
            lumaEndP = lumaAt(posP) - lumaAvg;
            doneP = abs(lumaEndP) >= gradient;
            distP = horzSpan ? (posP.x - posM.x) : (posP.y - posM.y);
        }
    }

    // ---- Seberapa jauh piksel ini dari ujung tepi terdekat? ---------------
    float dist = min(distN, distP);
    float spanLength = distN + distP;
    float pixelOffset = (-dist / max(spanLength, 1e-6)) + 0.5;

    // Penjagaan: kalau kedua ujung tepi berada di sisi luma yang SAMA dengan
    // piksel tengah, yang ditemukan bukan tepi melainkan puncak atau lembah.
    // Menggesernya justru akan merusak, jadi hanya koreksi subpiksel yang
    // dipakai. Tanpa penjagaan ini, garis tipis dan tekstur berpola berkedip.
    bool  lumaMLTAvg = lumaM < lumaAvg;
    bool  goodSpanN  = ((lumaEndN < 0.0) != lumaMLTAvg);
    bool  goodSpanP  = ((lumaEndP < 0.0) != lumaMLTAvg);
    bool  goodSpan   = (distN < distP) ? goodSpanN : goodSpanP;
    float subpixOffset = goodSpan ? pixelOffset : 0.0;

    // Yang terbesar antara geseran tepi dan koreksi subpiksel yang menang.
    //
    // v45: sebelumnya subpixH dikali 0,5 di sini. Itu SALAH — FXAA 3.11 asli
    // memakai max(pixelOffsetGood, subpixH) tanpa pengali. Akibatnya koreksi
    // subpiksel cuma bekerja setengah kekuatan, dan justru koreksi subpiksel
    // itulah yang menangani sliver 1-2 piksel seperti rusuk menara di scene
    // ini — fitur yang terlalu tipis untuk punya tepi panjang yang bisa
    // ditelusuri.
    float finalOffset = max(subpixOffset, subpixH);

    vec2 posOut = posM;
    if (horzSpan) posOut.y += finalOffset * lengthSign;
    else          posOut.x += finalOffset * lengthSign;

    return tonemapAt(posOut);
}

void main() {
    // ---- Penajaman, DI LUAR loop umpan balik TAA ---------------------------
    //
    // Dulu unsharp dikerjakan di taa_resolve, yang hasilnya ditulis ke history
    // dan dibaca lagi frame berikutnya — jadi penajamannya bertumpuk tiap frame
    // sampai meledak jadi bercak gelap. Di sini ia membaca history tapi tidak
    // pernah menulis kembali ke sana, jadi tidak ada yang bisa menumpuk.
    vec3 hdr = texture(hdrColor, vUV).rgb;

    if (pc.params.w > 0.0) {
        vec2 tx = 1.0 / vec2(textureSize(hdrColor, 0));
        vec3 blur = texture(hdrColor, vUV + vec2(-tx.x, 0.0)).rgb
                  + texture(hdrColor, vUV + vec2( tx.x, 0.0)).rgb
                  + texture(hdrColor, vUV + vec2(0.0, -tx.y)).rgb
                  + texture(hdrColor, vUV + vec2(0.0,  tx.y)).rgb;
        blur *= 0.25;

        vec3 sharp = hdr + (hdr - blur) * pc.params.w;

        // Penjagaan tambahan: hasil penajaman tidak boleh keluar dari rentang
        // tetangganya sendiri. Unsharp yang tak dibatasi bisa menembus nol dan
        // menghasilkan titik lebih gelap daripada apa pun di scene — bahkan
        // lebih gelap daripada bayangan penuh. Sekali kesalahan itu terjadi,
        // ia terlihat sebagai bercak yang tidak berhubungan dengan geometri.
        vec3 lo = min(hdr, blur), hi = max(hdr, blur);
        vec3 rng = hi - lo;
        hdr = clamp(sharp, lo - rng, hi + rng);
        hdr = max(hdr, vec3(0.0));
    }

    hdr *= pc.params.x;

    vec3 mapped = (pc.params.y > 0.5) ? acesFilm(hdr) : clamp(hdr, 0.0, 1.0);

    // Gamma HANYA kalau swapchain-nya bukan format sRGB.
    //
    // Engine ini meminta B8G8R8A8_SRGB dan biasanya mendapatkannya; pada format
    // sRGB, perangkat keras yang melakukan konversi saat menulis attachment.
    // Melakukannya lagi di sini akan membuat seluruh gambar pucat. Tapi
    // pemilihan format punya jalur cadangan `return f[0]` yang bisa
    // mengembalikan format non-sRGB, jadi C++ memberi tahu shader mana yang
    // sedang berlaku alih-alih menebak.
    if (pc.params.z > 0.5) mapped = pow(mapped, vec3(1.0 / 2.2));

    // ---- FXAA, PALING AKHIR -----------------------------------------------
    //
    // Urutannya wajib begini: FXAA bekerja pada nilai yang SUDAH ditonemap.
    // Menjalankannya di HDR akan membuat ambang kontrasnya tidak berarti —
    // selisih luma antara langit 4,0 dan dinding 0,3 jauh melampaui ambang
    // mana pun, jadi seluruh layar akan dianggap tepi.
    //
    // centerColor sengaja MEMBAWA hasil penajaman, sementara tap tetangga
    // tidak. Itu disengaja: penajaman butuh 4 fetch tambahan per tap, dan
    // FXAA memanggil sampai 29 tap — biayanya akan naik lima kali lipat untuk
    // beda yang tidak terlihat. Lagipula di 0,5 FPS pc.params.w bernilai nol
    // (ia dikali m_taaHealth), jadi di perangkat ini tidak ada beda sama
    // sekali.
    if (pc.fxaa.x >= 0.0) {
        vec2 rcp = 1.0 / vec2(textureSize(hdrColor, 0));

        // ---- MODE DEBUG (pc.fxaa.w) ---------------------------------------
        //
        // Ada di sini karena satu alasan: FXAA yang tidak bekerja TIDAK
        // menghasilkan error apa pun. Ia cuma diam. Tanpa cara melihat isi
        // kepalanya, satu-satunya cara menguji adalah menebak lalu build ulang.
        //
        //   1 = piksel yang LOLOS uji kontras dicat MAGENTA.
        //       Tidak ada magenta sama sekali = shader ini binari lama, atau
        //       pc.fxaa tidak sampai. Magenta di sepanjang siluet = FXAA jalan.
        //   2 = besar geseran akhir sebagai gradasi hijau.
        if (pc.fxaa.w > 0.5) {
            float lm = sqrt(dot(mapped, vec3(0.2126, 0.7152, 0.0722)));
            float ln = lumaAt(vUV + vec2(0.0, -rcp.y));
            float ls = lumaAt(vUV + vec2(0.0,  rcp.y));
            float le = lumaAt(vUV + vec2( rcp.x, 0.0));
            float lw = lumaAt(vUV + vec2(-rcp.x, 0.0));
            float rmax = max(lm, max(max(ln, ls), max(le, lw)));
            float rmin = min(lm, min(min(ln, ls), min(le, lw)));
            float rng  = rmax - rmin;
            bool  edge = rng >= max(pc.fxaa.z, rmax * pc.fxaa.y);

            if (pc.fxaa.w < 1.5) {
                outColor = edge ? vec4(1.0, 0.0, 1.0, 1.0)
                                : vec4(vec3(lm * 0.35), 1.0);
            } else {
                vec3 aa = fxaa(vUV, mapped, rcp);
                float moved = length(aa - mapped) * 6.0;
                outColor = vec4(0.0, clamp(moved, 0.0, 1.0), 0.0, 1.0);
            }
            return;
        }

        mapped = fxaa(vUV, mapped, rcp);
    }

    outColor = vec4(mapped, 1.0);
}

joystick.vert:

#version 450

layout(location = 0) in vec2 aPos;

layout(push_constant) uniform PC {
    vec4 uOffsetAndScale; // .xy = Offset (0,0 sekarang ada di Kiri Bawah), .z = Scale (Radius)
    vec4 uResolution;     // .xy = Resolusi Layar
    vec4 uColor;
} pc;

void main() {
    vec2 offset = pc.uOffsetAndScale.xy;
    float scale = pc.uOffsetAndScale.z;
    vec2 resolution = pc.uResolution.xy;

    // --- PERBAIKAN LOGIKA POSISI ---
    
    // 1. Hitung posisi dalam koordinat Layar (Pixel SDL)
    // Sumbu X: Offset bergerak dari kiri ke kanan
    float screenX = offset.x + (aPos.x * scale);

    // Sumbu Y: Offset bergerak dari bawah ke atas.
    // Di SDL, (0,0) ada di kiri atas. Titik paling bawah adalah `resolution.y`.
    // Jadi, untuk offset 0 berada di bawah, kita gunakan `resolution.y - offset.y`.
    // Kita juga kurangi `aPos.y * scale` agar bentuk joystick tidak terbalik secara vertikal.
    float screenY = (resolution.y - offset.y) - (aPos.y * scale);

    vec2 screenPos = vec2(screenX, screenY);

    // 2. Konversi dari Pixel (0..Width, 0..Height) ke Normalized Device Coordinates (-1..1)
    vec2 clipPos = (screenPos / resolution) * 2.0 - 1.0;

    // 3. Flip Y agar sesuai Vulkan (Y positif ke atas)
    // Pada langkah 2, jika screenY = resolution.y (bawah), maka clipPos.y = 1.
    // Di Vulkan, y=1 adalah Atas. Kita ingin Bawah (-1), jadi kita kalikan dengan -1.
    clipPos.y = -clipPos.y;

    gl_Position = vec4(clipPos, 0.0, 1.0);
}

joystick.frag:

#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec4 uOffsetAndScale;
    vec4 uResolution;
    vec4 uColor;
} pc;

void main() {
    outColor = pc.uColor;
}
