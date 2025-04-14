use super::AudioEffect;
use crate::config::CHANNELS;
// based on https://github.com/SergeiGorskiy/IFMO_Reverb

struct CombFilter {
    buffer: Vec<f32>,
    buffer_index: usize,
    feedback: f32,
    damping: f32,
    damp1: f32,
    damp2: f32,
}

struct AllPassFilter {
    buffer: Vec<f32>,
    buffer_index: usize,
    feedback: f32,
}

pub struct ReverbEffect {
    enabled: bool,
    pub room_size: f32,       // 0.0 to 1.0
    pub damping: f32,         // 0.0 to 1.0
    pub wet_level: f32,       // 0.0 to 1.0
    pub dry_level: f32,       // 0.0 to 1.0
    pub width: f32,           // 0.0 to 1.0
    pub freeze_mode: bool,
    
    // Internal state
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllPassFilter>,
}

impl CombFilter {
    fn new(size: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; size],
            buffer_index: 0,
            feedback,
            damping,
            damp1: 0.0,
            damp2: 1.0 - damping,
        }
    }
    
    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.buffer_index];
        
        self.damp1 = self.damp1 * self.damping + output * (1.0 - self.damping);
        
        let new_value = input + (self.damp1 * self.feedback);
        
        self.buffer[self.buffer_index] = new_value;
        
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();
        
        output
    }
    
    fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }
    
    fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
        self.damp2 = 1.0 - damping;
    }
    
    fn reset(&mut self) {
        for sample in self.buffer.iter_mut() {
            *sample = 0.0;
        }
    }
}

impl AllPassFilter {
    fn new(size: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; size],
            buffer_index: 0,
            feedback,
        }
    }
    
    fn process(&mut self, input: f32) -> f32 {
        let buffered_value = self.buffer[self.buffer_index];
        let output = -input + buffered_value;
        
        self.buffer[self.buffer_index] = input + (buffered_value * self.feedback);
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();
        
        output
    }
    
    fn reset(&mut self) {
        for sample in self.buffer.iter_mut() {
            *sample = 0.0;
        }
    }
}

impl ReverbEffect {
    pub fn new(sample_rate: u32) -> Self {
        // Constants for Schroeder-Moorer reverb algorithm
        let comb_tunings = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
            .iter()
            .map(|&v| (v as f32 * sample_rate as f32 / 44100.0) as usize)
            .collect::<Vec<_>>();
            
        let allpass_tunings = [556, 441, 341, 225]
            .iter()
            .map(|&v| (v as f32 * sample_rate as f32 / 44100.0) as usize)
            .collect::<Vec<_>>();
        
        let mut comb_filters = Vec::new();
        for &size in &comb_tunings {
            comb_filters.push(CombFilter::new(size, 0.84, 0.2));
        }
        
        let mut allpass_filters = Vec::new();
        for &size in &allpass_tunings {
            allpass_filters.push(AllPassFilter::new(size, 0.5));
        }
        
        Self {
            enabled: false,
            room_size: 0.5,
            damping: 0.5,
            wet_level: 0.33,
            dry_level: 0.4,
            width: 1.0,
            freeze_mode: false,
            comb_filters,
            allpass_filters,
        }
    }
    
    pub fn set_room_size(&mut self, room_size: f32) {
        self.room_size = room_size.max(0.0).min(1.0);
        self.update_comb_feedback();
    }
    
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.max(0.0).min(1.0);
        self.update_comb_damping();
    }
    
    pub fn set_wet_level(&mut self, wet_level: f32) {
        self.wet_level = wet_level.max(0.0).min(1.0);
    }
    
    pub fn set_dry_level(&mut self, dry_level: f32) {
        self.dry_level = dry_level.max(0.0).min(1.0);
    }
    
    pub fn set_width(&mut self, width: f32) {
        self.width = width.max(0.0).min(1.0);
    }
    
    pub fn set_freeze_mode(&mut self, freeze_mode: bool) {
        self.freeze_mode = freeze_mode;
        self.update_comb_feedback();
    }
    
    fn update_comb_feedback(&mut self) {
        let feedback = if self.freeze_mode {
            1.0
        } else {
            0.7 + 0.28 * self.room_size
        };
        
        for filter in &mut self.comb_filters {
            filter.set_feedback(feedback);
        }
    }
    
    fn update_comb_damping(&mut self) {
        for filter in &mut self.comb_filters {
            filter.set_damping(self.damping);
        }
    }
}

impl AudioEffect for ReverbEffect {
    fn process(&mut self, buffer: &mut [f32], _sample_rate: u32) {
        if !self.enabled || buffer.is_empty() {
            return;
        }
        
        let buffer_len = buffer.len();
        let num_frames = buffer_len / CHANNELS;
        
        for i in 0..num_frames {
            let base_idx = i * CHANNELS;
            
            // Get input samples
            let input_left = buffer[base_idx];
            let input_right = if CHANNELS > 1 { buffer[base_idx + 1] } else { input_left };
            
            let mut out_left = 0.0;
            let mut out_right = 0.0;
            
            // Process through comb filters
            for filter in &mut self.comb_filters {
                out_left += filter.process(input_left);
            }
            
            for filter in &mut self.comb_filters {
                out_right += filter.process(input_right);
            }
            
            // Process through allpass filters
            for filter in &mut self.allpass_filters {
                out_left = filter.process(out_left);
            }
            
            for filter in &mut self.allpass_filters {
                out_right = filter.process(out_right);
            }
            
            // Apply wet/dry mix
            let wet_gain = self.wet_level;
            let dry_gain = self.dry_level;
            
            buffer[base_idx] = input_left * dry_gain + out_left * wet_gain;
            
            if CHANNELS > 1 {
                buffer[base_idx + 1] = input_right * dry_gain + out_right * wet_gain;
            }
        }
    }

    fn reset(&mut self) {
        for filter in &mut self.comb_filters {
            filter.reset();
        }
        
        for filter in &mut self.allpass_filters {
            filter.reset();
        }
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.update_comb_feedback();
            self.update_comb_damping();
        }
    }
    
    fn name(&self) -> &str {
        "Reverb"
    }
}
