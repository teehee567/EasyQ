use super::AudioEffect;
use crate::config::CHANNELS;

pub struct DistortionEffect {
    enabled: bool,
    pub drive: f32,       // 1.0 to 100.0, amount of distortion
    pub tone: f32,        // 0.0 to 1.0, tone control (low-pass filter)
    pub mix: f32,         // 0.0 to 1.0, dry/wet mix
    pub output_gain: f32, // Output level compensation
    
    // Simple one-pole low-pass filter
    filter_state_left: f32,
    filter_state_right: f32,
}

impl DistortionEffect {
    pub fn new() -> Self {
        Self {
            enabled: false,
            drive: 10.0,
            tone: 0.5,
            mix: 0.5,
            output_gain: 0.5,
            filter_state_left: 0.0,
            filter_state_right: 0.0,
        }
    }
    
    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.max(1.0).min(100.0);
    }
    
    pub fn set_tone(&mut self, tone: f32) {
        self.tone = tone.max(0.0).min(1.0);
    }
    
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.max(0.0).min(1.0);
    }
    
    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain = gain.max(0.0).min(1.0);
    }
    
    fn apply_distortion(&self, sample: f32) -> f32 {
        let processed = (sample * self.drive).tanh();
        
        processed * (1.0 / (self.drive * 0.1).tanh())
    }
    
    fn apply_tone_filter(&mut self, sample: f32, is_left: bool) -> f32 {
        let alpha = self.tone.powf(2.0) * 0.99;
        
        let state = if is_left {
            &mut self.filter_state_left
        } else {
            &mut self.filter_state_right
        };
        
        *state = alpha * *state + (1.0 - alpha) * sample;
        
        *state
    }
}

impl AudioEffect for DistortionEffect {
    fn process(&mut self, buffer: &mut [f32], _sample_rate: u32) {
        if !self.enabled {
            return;
        }
        
        let buffer_len = buffer.len();
        let num_frames = buffer_len / CHANNELS;
        
        for i in 0..num_frames {
            let base_idx = i * CHANNELS;
            
            let in_left = buffer[base_idx];
            let in_right = if CHANNELS > 1 { buffer[base_idx + 1] } else { in_left };
            
            // Apply distortion
            let distorted_left = self.apply_distortion(in_left);
            let distorted_right = self.apply_distortion(in_right);
            
            // Apply tone filter
            let filtered_left = self.apply_tone_filter(distorted_left, true);
            let filtered_right = self.apply_tone_filter(distorted_right, false);
            
            // Apply wet/dry mix
            buffer[base_idx] = (in_left * (1.0 - self.mix) + filtered_left * self.mix) * self.output_gain;
            
            if CHANNELS > 1 {
                buffer[base_idx + 1] = (in_right * (1.0 - self.mix) + filtered_right * self.mix) * self.output_gain;
            }
        }
    }
    
    fn reset(&mut self) {
        self.filter_state_left = 0.0;
        self.filter_state_right = 0.0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    fn name(&self) -> &str {
        "Distortion"
    }
}
