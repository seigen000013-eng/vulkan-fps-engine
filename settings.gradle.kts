// v3: BLOK REPOSITORIES DITAMBAHKAN.
//
// Versi sebelumnya tidak punya ini sama sekali, dan itu kesalahan saya.
// Sejak Gradle 7, plugin dan dependensi TIDAK PUNYA repositori bawaan — kalau
// tidak disebutkan di sini, Gradle tidak tahu harus mengunduh
// com.android.application dari mana dan gagal saat fase konfigurasi, sebelum
// satu baris kode pun dikompilasi.
//
// Itu cocok dengan gejalanya: "BUILD FAILED in 23s" tanpa sempat memanggil
// compiler.
//
// google()            -> Android Gradle Plugin dan library AndroidX
// mavenCentral()      -> library umum
// gradlePluginPortal()-> plugin Gradle lain

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "VulkanFPS"

include(":app")

// CATATAN: modul ":SDL3" sengaja TIDAK disertakan.
//
// android-project/app milik SDL3 adalah modul APLIKASI lengkap — ia menerapkan
// plugin com.android.application sendiri, dan satu build Gradle tidak boleh
// punya dua modul aplikasi.
//
// Sumber Java SDL (org/libsdl/app) disalin langsung ke app/src/main/java oleh
// workflow, jadi ia dikompilasi sebagai bagian dari modul app.
