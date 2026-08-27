rootProject.name = "VulkanFPS"

include(":app")

// Modul Java SDL3. Sumbernya ada di dalam rilis SDL3, di
// android-project/app/src/main/java — tapi Gradle butuh modul tersendiri.
include(":SDL3")
project(":SDL3").projectDir = file("app/src/main/cpp/SDL3/android-project/app")
