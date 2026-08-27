plugins { id("com.android.application") }

android {
    namespace  = "com.vulkanfps.engine"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.vulkanfps.engine"

        // 26 = Android 8.0. Vulkan baru benar-benar bisa diandalkan sejak sini.
        minSdk    = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            // HANYA arm64. Membangun untuk armeabi-v7a menggandakan waktu build
            // dan ukuran APK untuk arsitektur yang tidak kamu pakai.
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
            // Debug symbol dibuang dari APK rilis; ukurannya turun drastis.
            ndk { debugSymbolLevel = "NONE" }
        }
    }

    // .spv sudah terkompresi buruk dan HARUS bisa dibaca apa adanya.
    // Tanpa baris ini, SDL_LoadFile tetap jalan, tapi tiap pemuatan shader
    // harus melewati dekompresi — sia-sia untuk berkas biner.
    androidResources { noCompress += listOf("spv", "obj") }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // SDL3 sisi Java. Path relatif ke tempat kamu menaruh sumber SDL3.
    implementation(project(":SDL3"))
}
