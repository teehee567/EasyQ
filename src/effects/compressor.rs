use super::AudioEffect;
use crate::config::CHANNELS;

pub struct CompressorEffect {
    enabled: bool,
    pub threshold: f32,      // -60.0 to 0.0 dB, the level at which compression begins
    pub ratio: f32,          // 1.0 to 20.0, amount of compression applied
    pub attack: f32,         // 0.1 to 200.0 ms, how quickly compression is applied
    pub release: f32,        // 10.0 to 2000.0 ms, how quickly compression is released
    pub makeup_gain: f32,    // 0.0 to 24.0 dB, compensates for volume reduction
    
    envelope_left: f32,
    envelope_right: f32,
}

impl CompressorEffect {
    pub fn new() -> Self {
        Self {
            enabled: false,
            threshold: -20.0,
            ratio: 4.0,
            attack: 20.0,
            release: 200.0,
            makeup_gain: 0.0,
            envelope_left: 0.0,
            envelope_right: 0.0,
        }
    }
    
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(-60.0).min(0.0);
    }
    
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.max(1.0).min(20.0);
    }
    
    pub fn set_attack(&mut self, attack_ms: f32) {
        self.attack = attack_ms.max(0.1).min(200.0);
    }
    
    pub fn set_release(&mut self, release_ms: f32) {
        self.release = release_ms.max(10.0).min(2000.0);
    }
    
    pub fn set_makeup_gain(&mut self, gain_db: f32) {
        self.makeup_gain = gain_db.max(0.0).min(24.0);
    }
    
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }
    
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.abs().max(1e-6).log10()
    }
    
    fn calculate_gain_reduction(&self, envelope_db: f32) -> f32 {
        if envelope_db <= self.threshold {
            return 0.0;
        }
        
        let above_threshold = envelope_db - self.threshold;
        
        let attenuation = above_threshold - (above_threshold / self.ratio);
        
        -attenuation
    }
    
    fn update_envelope(&mut self, input: f32, is_left: bool, sample_rate: u32) -> f32 {
        let envelope = if is_left {
            &mut self.envelope_left
        } else {
            &mut self.envelope_right
        };
        
        let input_abs = input.abs();
        
        let attack_coeff = (-2.2 / (self.attack / 1000.0 * sample_rate as f32)).exp();
        let release_coeff = (-2.2 / (self.release / 1000.0 * sample_rate as f32)).exp();
        
        if input_abs > *envelope {
            *envelope = attack_coeff * (*envelope) + (1.0 - attack_coeff) * input_abs;
        } else {
            *envelope = release_coeff * (*envelope) + (1.0 - release_coeff) * input_abs;
        }
        
        *envelope
    }
}

impl AudioEffect for CompressorEffect {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32) {
        if !self.enabled {
            return;
        }
        
        let buffer_len = buffer.len();
        let num_frames = buffer_len / CHANNELS;
        
        let makeup_gain_linear = Self::db_to_linear(self.makeup_gain);
        
        for i in 0..num_frames {
            let base_idx = i * CHANNELS;
            
            let in_left = buffer[base_idx];
            let in_right = if CHANNELS > 1 { buffer[base_idx + 1] } else { in_left };
            
            // Update envelope followers
            let envelope_left = self.update_envelope(in_left, true, sample_rate);
            let envelope_right = if CHANNELS > 1 {
                self.update_envelope(in_right, false, sample_rate)
            } else {
                envelope_left
            };
            
            let envelope_db_left = Self::linear_to_db(envelope_left);
            let envelope_db_right = Self::linear_to_db(envelope_right);
            
            // Calculate gain reduction
            let gain_reduction_db_left = self.calculate_gain_reduction(envelope_db_left);
            let gain_reduction_db_right = self.calculate_gain_reduction(envelope_db_right);
            
            let gain_left = Self::db_to_linear(gain_reduction_db_left) * makeup_gain_linear;
            let gain_right = Self::db_to_linear(gain_reduction_db_right) * makeup_gain_linear;
            
            // Apply gain reduction
            buffer[base_idx] = in_left * gain_left;
            
            if CHANNELS > 1 {
                buffer[base_idx + 1] = in_right * gain_right;
            }
        }
    }
    
    fn reset(&mut self) {
        self.envelope_left = 0.0;
        self.envelope_right = 0.0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    fn name(&self) -> &str {
        "Compressor"
    }
}
