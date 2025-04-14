use super::AudioEffect;
use crate::config::CHANNELS;

pub struct DelayEffect {
    enabled: bool,
    pub delay_time_ms: f32,      // Delay time in milliseconds
    pub feedback: f32,           // 0.0 to 1.0, how much of the delayed signal is fed back
    pub mix: f32,                // 0.0 to 1.0, dry/wet mix
    buffer_left: Vec<f32>,       // Circular buffer for left channel
    buffer_right: Vec<f32>,      // Circular buffer for right channel
    buffer_size: usize,          // Size of buffer based on max delay time
    write_pos: usize,            // Current write position in buffer
    sample_rate: u32,            // Sample rate for calculating delay time
}

impl DelayEffect {
    pub fn new(sample_rate: u32) -> Self {
        let max_delay_ms = 2000.0;
        let buffer_size = ((sample_rate as f32 * max_delay_ms / 1000.0) as usize).next_power_of_two();
        
        Self {
            enabled: false,
            delay_time_ms: 500.0, // 500ms default
            feedback: 0.3,        // 30% feedback
            mix: 0.5,             // 50/50 mix
            buffer_left: vec![0.0; buffer_size],
            buffer_right: vec![0.0; buffer_size],
            buffer_size,
            write_pos: 0,
            sample_rate,
        }
    }
    
    pub fn set_delay_time(&mut self, delay_time_ms: f32) {
        self.delay_time_ms = delay_time_ms.max(10.0).min(2000.0);
    }
    
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.max(0.0).min(0.95);
    }
    
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.max(0.0).min(1.0);
    }
    
    fn get_read_pos(&self) -> usize {
        let delay_samples = (self.sample_rate as f32 * self.delay_time_ms / 1000.0) as usize;
        (self.write_pos + self.buffer_size - delay_samples) % self.buffer_size
    }
}

impl AudioEffect for DelayEffect {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32) {
        if !self.enabled {
            return;
        }
        
        if self.sample_rate != sample_rate {
            self.sample_rate = sample_rate;
        }
        
        let buffer_len = buffer.len();
        let num_frames = buffer_len / CHANNELS;
        
        for i in 0..num_frames {
            let base_idx = i * CHANNELS;
            
            let in_left = buffer[base_idx];
            let in_right = if CHANNELS > 1 { buffer[base_idx + 1] } else { in_left };
            
            // Calculate read position
            let read_pos = self.get_read_pos();
            
            // Read delayed samples
            let delayed_left = self.buffer_left[read_pos];
            let delayed_right = self.buffer_right[read_pos];
            
            // Calculate new samples with feedback
            let new_left = in_left + delayed_left * self.feedback;
            let new_right = in_right + delayed_right * self.feedback;
            
            // Write to buffer
            self.buffer_left[self.write_pos] = new_left;
            self.buffer_right[self.write_pos] = new_right;
            
            // Update write position
            self.write_pos = (self.write_pos + 1) % self.buffer_size;
            
            // Mix dry and wet signals
            buffer[base_idx] = in_left * (1.0 - self.mix) + delayed_left * self.mix;
            
            if CHANNELS > 1 {
                buffer[base_idx + 1] = in_right * (1.0 - self.mix) + delayed_right * self.mix;
            }
        }
    }
    
    fn reset(&mut self) {
        for i in 0..self.buffer_size {
            self.buffer_left[i] = 0.0;
            self.buffer_right[i] = 0.0;
        }
        self.write_pos = 0;
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    fn name(&self) -> &str {
        "Delay"
    }
}
