use core::f32;

use crate::config::{BANDS, FREQUENCIES};

struct BiquadFilter {
    b0: f32, b1: f32, b2: f32, // Numerator coefficients
    a1: f32, a2: f32, // Denominator coefficients
    
    // State variables for the filter
    x1: f32, x2: f32, // Previous inputs
    y1: f32, y2: f32, // Previous outputs
}

impl BiquadFilter {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        BiquadFilter {
            b0, b1, b2, a1, a2,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
        }
    }
    
    fn process(&mut self, input: f32) -> f32 {
        // Direct Form 1 implementation
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
                   - self.a1 * self.y1 - self.a2 * self.y2;
        
        // Update state
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        
        output
    }
    
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

fn calculate_bandwidth(center_freq: f32, neighbor_freq: Option<f32>) -> f32 {
    match neighbor_freq {
        Some(next_freq) => {
            // Calculate geometric mean between this frequency and the next
            let ratio = (next_freq / center_freq).sqrt();
            (ratio - 1.0/ratio) / 1.4142
        },
        None => 1.0
    }
}

pub struct EQProcessor {
    filters: Vec<BiquadFilter>,
    current_gains: [f32; BANDS],
    sample_rate: f32,
    needs_update: bool,
}

impl EQProcessor {
    pub fn new(sample_rate: u32) -> Self {
        let mut processor = Self {
            filters: Vec::with_capacity(BANDS),
            current_gains: [0.0; BANDS],
            sample_rate: sample_rate as f32,
            needs_update: true,
        };
        
        processor.update_filters();
        
        processor
    }
    
fn update_filters(&mut self) {
    self.filters.clear();
    
    for i in 0..FREQUENCIES.len() {
        let freq = FREQUENCIES[i];
        let gain_db = self.current_gains[i];
        
        // Skip processing for bands with effectively zero gain
        if gain_db.abs() < 0.01 {
            self.filters.push(BiquadFilter::new(1.0, 0.0, 0.0, 0.0, 0.0));
            continue;
        }
        
        // For the highest frequency band (20kHz), use a high-shelf filter instead of peaking
        if i == FREQUENCIES.len() - 1 {
            let (b0, b1, b2, a1, a2) = high_shelf_coefficients(
                self.sample_rate,
                freq,
                gain_db
            );
            self.filters.push(BiquadFilter::new(b0, b1, b2, a1, a2));
        } else {
            // For all other frequencies, use peaking filters with calculated bandwidths
            let next_freq = if i < FREQUENCIES.len() - 1 { Some(FREQUENCIES[i + 1]) } else { None };
            let bandwidth = calculate_bandwidth(freq, next_freq);
            let q = 1.0 / bandwidth;
            
            let (b0, b1, b2, a1, a2) = peaking_eq_coefficients(
                self.sample_rate,
                freq,
                q,
                gain_db
            );
            
            self.filters.push(BiquadFilter::new(b0, b1, b2, a1, a2));
        }
    }
    
    self.needs_update = false;
}

    
    pub fn update_gains(&mut self, new_gains: [f32; BANDS]) {
        if self.current_gains != new_gains {
            self.current_gains = new_gains;
            self.needs_update = true;
        }
    }
    
    pub fn update_sample_rate(&mut self, sample_rate: u32) {
        let sample_rate = sample_rate as f32;
        if (self.sample_rate - sample_rate).abs() > 0.1 {
            self.sample_rate = sample_rate;
            self.needs_update = true;
            
            for filter in &mut self.filters {
                filter.reset();
            }
        }
    }
    
    pub fn process(&mut self, sample: f32) -> f32 {
        // Check if filters need to be updated
        if self.needs_update {
            self.update_filters();
        }
        
        let mut output = sample;
        for filter in &mut self.filters {
            output = filter.process(output);
        }
        output
    }
}


fn high_shelf_coefficients(sample_rate: f32, frequency: f32, gain_db: f32) -> (f32, f32, f32, f32, f32) {
    // Convert gain from dB to linear amplitude
    let amp = 10.0f32.powf(gain_db / 20.0);
    
    // Calculate intermediate values
    let omega = 2.0 * f32::consts::PI * frequency / sample_rate;
    let alpha = omega.sin() / 2.0 * (2.0f32).sqrt();
    let cos_omega = omega.cos();
    
    // Calculate filter coefficients (based on Audio EQ Cookbook)
    let b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_omega + 2.0 * amp.sqrt() * alpha);
    let b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_omega);
    let b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_omega - 2.0 * amp.sqrt() * alpha);
    let a0 = (amp + 1.0) - (amp - 1.0) * cos_omega + 2.0 * amp.sqrt() * alpha;
    let a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_omega);
    let a2 = (amp + 1.0) - (amp - 1.0) * cos_omega - 2.0 * amp.sqrt() * alpha;
    
    (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)
}

fn peaking_eq_coefficients(sample_rate: f32, frequency: f32, q: f32, gain_db: f32) -> (f32, f32, f32, f32, f32) {
    // Convert gain from dB to linear amplitude
    let amp = 10.0f32.powf(gain_db / 40.0);
    
    // Calculate intermediate values with safety check for alpha
    let omega = 2.0 * f32::consts::PI * frequency / sample_rate;
    let alpha = (omega.sin() / (2.0 * q)).max(0.001); // Prevent alpha from being too small
    let cos_omega = omega.cos();
    
    // Calculate filter coefficients (based on Audio EQ Cookbook)
    let b0 = 1.0 + alpha * amp;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * amp;
    let a0 = 1.0 + alpha / amp;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / amp;
    
    // Normalize by a0
    (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)
}
