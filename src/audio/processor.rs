use rustfft::{FftPlanner, num_complex::{Complex, Complex32}};
use apodize::hanning_iter;
use crate::config::{CHANNELS, FFT_SIZE, SPECTRUM_POINTS, OVERLAP, SPECTRUM_HISTORY};

pub struct SpectrumAnalyzer {
    pub fft_planner: FftPlanner<f32>,
    pub fft_input: Vec<Complex32>,
    pub fft_output: Vec<Complex32>,
    pub window: Vec<f32>,
    pub buffer: Vec<f32>,
    pub buffer_pos: usize,
    pub spectrum_history: Vec<Vec<f32>>,
    pub freq_bins: Vec<usize>,
}

impl SpectrumAnalyzer {
    pub fn new() -> Self {
        let mut spectrum_history = Vec::with_capacity(SPECTRUM_HISTORY);
        for _ in 0..SPECTRUM_HISTORY {
            spectrum_history.push(vec![0.0; SPECTRUM_POINTS]);
        }
        
        let freq_bins = Self::create_log_freq_bins(SPECTRUM_POINTS, FFT_SIZE / 2);
        
        Self {
            fft_planner: FftPlanner::new(),
            fft_input: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            fft_output: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window: hanning_iter(FFT_SIZE).map(|x| x as f32).collect(),
            buffer: vec![0.0; FFT_SIZE],
            buffer_pos: 0,
            spectrum_history,
            freq_bins,
        }
    }

    fn create_log_freq_bins(num_points: usize, max_bin: usize) -> Vec<usize> {
        let min_freq = 20f32;
        let max_freq = 20000f32;
        
        let min_log = min_freq.ln();
        let max_log = max_freq.ln();
        
        let mut bins = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let t = i as f32 / (num_points - 1) as f32;
            let log_freq = min_log + t * (max_log - min_log);
            let freq = log_freq.exp();
            
            // Convert frequency to bin index
            let bin = ((freq / max_freq) * max_bin as f32).round() as usize;
            bins.push(bin.min(max_bin - 1).max(0));
        }
        
        bins
    }

    pub fn add_samples(&mut self, samples: &[f32]) -> bool {
        let mut fft_ready = false;
        
        for chunk in samples.chunks(CHANNELS) {
            if chunk.len() == CHANNELS {
                let sample = chunk.iter().sum::<f32>() / (CHANNELS as f32);
                
                self.buffer[self.buffer_pos] = sample;
                self.buffer_pos += 1;
                
                if self.buffer_pos >= self.buffer.len() {
                    fft_ready = true;
                    
                    if OVERLAP > 0 && OVERLAP < FFT_SIZE {
                        for i in 0..OVERLAP {
                            self.buffer[i] = self.buffer[FFT_SIZE - OVERLAP + i];
                        }
                        self.buffer_pos = OVERLAP;
                    } else {
                        self.buffer_pos = 0;
                    }
                }
            }
        }
        
        fft_ready
    }
    
    pub fn process_fft(&mut self, sample_rate: u32) -> Vec<f32> {
        for i in 0..FFT_SIZE {
            self.fft_input[i] = Complex32::new(self.buffer[i] * self.window[i], 0.0);
        }
        
        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        self.fft_output.copy_from_slice(&self.fft_input);
        fft.process(&mut self.fft_output);
        
        let mut spectrum = vec![0.0; SPECTRUM_POINTS];
        let scale_factor = 2.0 / (FFT_SIZE as f32);
        
        // Calculate magnitudes for each logarithmic bin
        for (i, &bin) in self.freq_bins.iter().enumerate() {
            let start_bin = if bin > 0 { bin - 1 } else { bin };
            let end_bin = (bin + 1).min(FFT_SIZE / 2 - 1);
            
            let mut sum_energy = 0.0;
            let mut count = 0;
            
            for j in start_bin..=end_bin {
                let mag = self.fft_output[j].norm() * scale_factor;
                sum_energy += mag;
                count += 1;
            }
            
            let avg_energy = if count > 0 { sum_energy / count as f32 } else { 0.0 };
            
            // Convert to dB scale
            let db = 20.0 * (avg_energy + 1e-10).log10();
            
            // Normalize to 0-60 range with floor at -80dB
            spectrum[i] = (db + 80.0).max(0.0).min(60.0);
        }
        
        self.spectrum_history.remove(0);
        self.spectrum_history.push(spectrum.clone());
        
        self.average_spectrum()
    }
    
    // Calculate average spectrum from history
    fn average_spectrum(&self) -> Vec<f32> {
        let mut avg_spectrum = vec![0.0; SPECTRUM_POINTS];
        
        for i in 0..SPECTRUM_POINTS {
            // Calculate weighted average, giving more weight to recent frames
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for (idx, frame) in self.spectrum_history.iter().enumerate() {
                // Weight increases with index (newer frames have higher weight)
                let weight = (idx + 1) as f32;
                sum += frame[i] * weight;
                weight_sum += weight;
            }
            
            avg_spectrum[i] = if weight_sum > 0.0 { sum / weight_sum } else { 0.0 };
        }
        
        avg_spectrum
    }
    
    // Helper method to get frequency for a specific bin (useful for debugging)
    pub fn get_frequency_for_bin(&self, bin: usize, sample_rate: u32) -> f32 {
        if bin < self.freq_bins.len() {
            let fft_bin = self.freq_bins[bin];
            fft_bin as f32 * sample_rate as f32 / FFT_SIZE as f32
        } else {
            0.0
        }
    }
}
