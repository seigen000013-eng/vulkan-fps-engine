#!/usr/bin/env bash
# =============================================================================
#  compile_shaders.sh — ekstrak GLSL dari shaders.cpp, resolusi #include,
#                       compile ke SPIR-V
#  Target: main2.cpp + Shadow AAA
# =============================================================================
#  Butuh : glslc (Vulkan SDK / shaderc) ATAU glslangValidator
#  Pakai : ./compile_shaders.sh [file_sumber] [dir_output]
# =============================================================================
#  DUA PERUBAHAN DARI VERSI SEBELUMNYA:
#
#  1. Blok berakhiran .glsl diperlakukan sebagai INCLUDE, bukan shader. Ia
#     diekstrak tapi tidak dikompilasi. Baris  #include "nama.glsl"  di dalam
#     shader disisipkan apa adanya sebelum dikirim ke compiler. Ini menghapus
#     duplikasi ShadowUBO yang tadinya tersalin ke lima shader — 110 baris yang
#     harus selalu sinkron, dan yang kalau meleset BUKAN compile error melainkan
#     pergeseran layout std140 yang muncul sebagai nilai acak saat runtime.
#
#  2. Daftar compile TIDAK LAGI DITULIS TANGAN. Ia diturunkan dari blok yang
#     benar-benar ada di file sumber. Versi lama menyimpan daftar manual, dan
#     komentarnya sendiri sudah mengakui bahayanya: menambah shader tanpa
#     menambah barisnya lolos diam-diam, lalu gagal sebagai crash readFile()
#     saat aplikasi jalan. Sekarang mustahil lupa.
# =============================================================================
set -euo pipefail

SRC="${1:-shaders.cpp}"
OUT="${2:-.}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# main2.cpp membuat instance dengan VK_API_VERSION_1_3, jadi target-env-nya
# harus ikut 1.3 — kalau lebih rendah, validation layer akan protes soal
# SPIR-V environment mismatch.
VK_ENV="vulkan1.3"

if [ ! -f "$SRC" ]; then
    echo "ERROR: file tidak ditemukan: $SRC" >&2
    exit 1
fi

# --- pilih compiler -----------------------------------------------------------
if command -v glslc >/dev/null 2>&1; then
    COMPILER="glslc"
elif command -v glslangValidator >/dev/null 2>&1; then
    COMPILER="glslang"
else
    echo "ERROR: glslc atau glslangValidator tidak ditemukan." >&2
    echo "       Install Vulkan SDK: https://vulkan.lunarg.com/" >&2
    exit 1
fi

echo ">> compiler : $COMPILER"
echo ">> target   : $VK_ENV"
echo ">> sumber   : $SRC"
echo ">> output   : $OUT"
mkdir -p "$OUT"

# --- ekstrak blok ke file terpisah --------------------------------------------
# Judul blok harus di kolom 1 dan diakhiri titik dua, mis. 'csm_resolve.comp:'
# atau 'common_ubo.glsl:'. Header komentar di awal shaders.cpp tidak ikut
# terekstrak karena `out` masih kosong sampai judul blok pertama ketemu.
echo ">> ekstraksi..."
awk -v dir="$TMP" '
    /^[A-Za-z0-9_]+\.(vert|frag|comp|glsl):[[:space:]]*$/ {
        name = $0
        sub(/:[[:space:]]*$/, "", name)
        out  = dir "/" name
        skip = 1
        printf "   extract  %s\n", name > "/dev/stderr"
        next
    }
    skip && /^[[:space:]]*$/ { next }
    out != "" { skip = 0; print > out }
' "$SRC"

if [ -z "$(ls -A "$TMP")" ]; then
    echo "ERROR: tidak ada blok yang terekstrak dari $SRC" >&2
    echo "       Judul blok harus di kolom 1 dan diakhiri titik dua." >&2
    exit 1
fi

# --- resolusi #include --------------------------------------------------------
# Disisipkan secara tekstual di sini, BUKAN diserahkan ke -I milik glslc, supaya
# perilakunya sama persis di glslc maupun glslangValidator. Diulang sampai tidak
# ada lagi baris #include, sehingga include bersarang ikut tertangani; batas 5
# putaran menangkap include yang saling memanggil.
resolve_includes() {
    f="$1"
    pass=0
    while grep -q '^[[:space:]]*#include[[:space:]]*"' "$f"; do
        pass=$((pass + 1))
        if [ "$pass" -gt 5 ]; then
            echo "   ERROR: #include melingkar di '$(basename "$f")'" >&2
            exit 1
        fi
        awk -v dir="$TMP" '
            /^[[:space:]]*#include[[:space:]]*"/ {
                name = $0
                sub(/^[[:space:]]*#include[[:space:]]*"/, "", name)
                sub(/".*$/, "", name)
                path = dir "/" name
                if ((getline probe < path) < 0) {
                    printf "   ERROR: include tidak ditemukan: %s\n", name > "/dev/stderr"
                    exit 1
                }
                close(path)
                printf "// ---- awal %s ----\n", name
                while ((getline line < path) > 0) print line
                close(path)
                printf "// ---- akhir %s ----\n", name
                next
            }
            { print }
        ' "$f" > "$f.inc"
        mv "$f.inc" "$f"
    done
}

echo ">> resolusi #include..."
for f in "$TMP"/*.vert "$TMP"/*.frag "$TMP"/*.comp; do
    [ -e "$f" ] || continue
    resolve_includes "$f"
done

# --- helper kompilasi ---------------------------------------------------------
COMPILED=""

compile() {
    f="$1"; stage="$2"; spv="$3"

    # Blok yang terekstrak setengah (mis. judul salah ketik sehingga isi blok
    # sebelumnya ikut tertulis) paling cepat ketahuan dari baris #version.
    if ! head -5 "$f" | grep -q '^#version'; then
        echo "   ERROR: '$(basename "$f")' tidak diawali #version." >&2
        echo "          Kemungkinan judul blok di $SRC salah tulis." >&2
        exit 1
    fi

    # Pemeriksa identifier: menangkap variabel yang DIPAKAI tapi tidak pernah
    # DIDEKLARASIKAN. Dijalankan sebelum glslc supaya pesannya menyebut nama
    # variabelnya, bukan cuma nomor baris.
    if [ -f "$(dirname "$0")/check_glsl.py" ] && command -v python3 >/dev/null 2>&1; then
        python3 "$(dirname "$0")/check_glsl.py" "$f" >/dev/null 2>&1 || {
            python3 "$(dirname "$0")/check_glsl.py" "$f" >&2
            echo "          (periksa $SRC pada blok '$(basename "$f")')" >&2
            exit 1
        }
    fi

    if [ "$COMPILER" = "glslc" ]; then
        glslc --target-env="$VK_ENV" -fshader-stage="$stage" -O "$f" -o "$OUT/$spv"
    else
        # glslangValidator memakai nama stage pendek (vert/frag/comp), BUKAN
        # vertex/fragment/compute — kalau dikirim yang panjang, ia gagal.
        case "$stage" in
            vertex)   gstage="vert" ;;
            fragment) gstage="frag" ;;
            compute)  gstage="comp" ;;
            *)        gstage="$stage" ;;
        esac
        glslangValidator -V --target-env "$VK_ENV" -S "$gstage" "$f" -o "$OUT/$spv"
    fi
    COMPILED="$COMPILED $(basename "$f")"
    echo "   OK   $spv"
}

# --- kompilasi: daftar DITURUNKAN dari blok yang ada --------------------------
# Blok .glsl sengaja tidak masuk glob di bawah: ia bahan include, bukan shader.
echo ">> kompilasi..."
for f in "$TMP"/*.vert "$TMP"/*.frag "$TMP"/*.comp; do
    [ -e "$f" ] || continue
    n="$(basename "$f")"
    case "$n" in
        *.vert) stage="vertex"   ;;
        *.frag) stage="fragment" ;;
        *.comp) stage="compute"  ;;
        *)      continue         ;;
    esac
    compile "$f" "$stage" "$n.spv"
done

# --- penjagaan: .spv yang dicari main2.cpp tapi tidak ada --------------------
# Sumber crash paling sering adalah nama file yang tidak sinkron antara
# shaders.cpp dan string literal di main2.cpp. Daftar ini SATU-SATUNYA yang
# masih ditulis tangan, dan memang harus begitu: ia mencerminkan apa yang
# DIMINTA main2.cpp, bukan apa yang kebetulan ada di shaders.cpp. Justru karena
# itu ia tetap berguna — ia menangkap ketidaksinkronan antara kedua file.
REQUIRED="
prepass.vert.spv prepass.frag.spv
csm_shadow.vert.spv csm_shadow.frag.spv
ao_resolve.comp.spv
volumetric.comp.spv
csm_resolve.comp.spv
shadow_temporal.comp.spv
svgf_atrous.comp.spv
main.vert.spv main.frag.spv
sky.vert.spv sky.frag.spv
taa_resolve.comp.spv
composite.vert.spv composite.frag.spv
joystick.vert.spv joystick.frag.spv
"
NOTFOUND=""
for r in $REQUIRED; do
    [ -f "$OUT/$r" ] || NOTFOUND="$NOTFOUND $r"
done

if [ -n "$NOTFOUND" ]; then
    echo "" >&2
    echo "ERROR: .spv yang dibutuhkan main2.cpp tidak terbentuk:" >&2
    for n in $NOTFOUND; do echo "   - $n" >&2; done
    echo "       Periksa apakah blok shader-nya ada di $SRC." >&2
    exit 1
fi

echo ""
echo ">> selesai. $(echo $COMPILED | wc -w) file .spv ada di: $OUT"
