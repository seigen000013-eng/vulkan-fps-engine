plugins { id("com.android.application") }

android {
    namespace  = "com.vulkanfps.engine"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.vulkanfps.engine"
        minSdk    = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            // HANYA arm64 — armeabi-v7a cuma menggandakan waktu build.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                // c++_shared WAJIB: SDL3 dan engine dua .so terpisah, dan
                // keduanya harus memakai runtime C++ yang SAMA. Dengan
                // c++_static, exception yang dilempar di satu .so tidak bisa
                // ditangkap di .so lain — dan main2.cpp memakai try/catch.
                arguments += listOf("-DANDROID_STL=c++_shared")
                cppFlags  += listOf("-std=c++20", "-O2", "-fexceptions", "-frtti")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            ndk { debugSymbolLevel = "NONE" }
        }
    }

    // .spv harus bisa dibaca apa adanya, tanpa melewati dekompresi.
    androidResources { noCompress += listOf("spv", "obj") }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// Tidak ada blok dependencies{}. Sumber Java SDL3 disalin langsung ke
// app/src/main/java oleh workflow, jadi ia dikompilasi sebagai bagian dari
// modul ini — bukan sebagai modul Gradle terpisah.
