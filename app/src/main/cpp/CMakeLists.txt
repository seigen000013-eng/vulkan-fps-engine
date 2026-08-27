cmake_minimum_required(VERSION 3.22.1)
project(vulkanfps LANGUAGES CXX)

# ---------------------------------------------------------------------------
# SDL3 — dibangun dari sumber sebagai bagian dari proyek ini.
#
# Taruh sumber SDL3 di app/src/main/cpp/SDL3 (git clone atau salin rilisnya).
# Membangunnya dari sumber lebih andal daripada memakai .so prebuilt, karena
# versi header dan biner dijamin cocok.
# ---------------------------------------------------------------------------
add_subdirectory(SDL3)

# ---------------------------------------------------------------------------
# GLM — header-only, cukup ditunjuk direktorinya.
# ---------------------------------------------------------------------------
add_library(glm_headers INTERFACE)
target_include_directories(glm_headers INTERFACE glm)

# ---------------------------------------------------------------------------
# Engine
#
# NAMANYA HARUS "vulkanfps" — cocok dengan getLibraries() di EngineActivity.
# Kalau tidak cocok, APK terpasang tapi langsung crash saat start dengan
# UnsatisfiedLinkError.
# ---------------------------------------------------------------------------
add_library(vulkanfps SHARED main2.cpp)

target_link_libraries(vulkanfps
    PRIVATE
        SDL3::SDL3
        glm_headers
        vulkan          # libvulkan.so milik Android, bukan Mesa
        android
        log)

target_compile_definitions(vulkanfps PRIVATE
    # vulkan.hpp memakai dynamic dispatch; ini mencegahnya menuntut simbol
    # yang tidak ada di loader Android.
    VULKAN_HPP_NO_EXCEPTIONS_ON_MISSING_SYMBOLS=1)
