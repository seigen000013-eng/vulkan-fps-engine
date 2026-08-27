package com.vulkanfps.engine;

import org.libsdl.app.SDLActivity;

/**
 * Seluruh siklus hidup Android ditangani SDLActivity: pembuatan surface, input
 * sentuh, pause/resume, dan pemanggilan main() di sisi native.
 *
 * getLibraries() menentukan URUTAN pemuatan .so, dan urutannya penting: SDL3
 * harus sudah termuat sebelum biner engine, karena engine menautnya.
 */
public class EngineActivity extends SDLActivity {
    @Override
    protected String[] getLibraries() {
        return new String[] { "SDL3", "vulkanfps" };
    }
}
