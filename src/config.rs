pub const BANDS: usize = 11;
pub const FREQUENCIES: [f32; BANDS] = [
    31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
];
pub const BUFFER_SIZE: usize = 2048;
pub const CHANNELS: usize = 2;

pub const SPECTRUM_POINTS: usize = 512;
pub const SPECTRUM_HISTORY: usize = 3;
pub const SPECTRUM_DECAY: f32 = 0.7;

pub const FFT_SIZE: usize = 2048;
pub const OVERLAP: usize = FFT_SIZE / 2;
