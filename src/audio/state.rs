use std::collections::HashMap;
use crate::config::{BANDS, SPECTRUM_POINTS};
use crate::effects::eqprocessor::EQProcessor;
use crate::effects::EffectsChain;

pub struct AudioState {
    pub running: bool,
    pub eq_gains: [f32; BANDS],
    pub input_device_index: usize,
    pub output_device_index: usize,
    pub devices: Vec<String>,
    pub presets: HashMap<String, [f32; BANDS]>,
    pub eq_processor: EQProcessor,
    pub sample_rate: u32,
    pub spectrum_data: Vec<f32>,
    pub spectrum_peak: Vec<f32>,
    pub effects_chain: EffectsChain,
}

impl Default for AudioState {
    fn default() -> Self {
        let presets = HashMap::from([
            ("Flat".to_string(), [0.0; BANDS]),
            (
                "Bass Boost".to_string(),
                [7.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "Treble Boost".to_string(),
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 8.0],
            ),
            (
                "V-Shape".to_string(),
                [5.0, 3.0, 1.0, -1.0, -2.0, -2.0, -1.0, 1.0, 3.0, 5.0, 7.0],
            ),
            (
                "Vocal Boost".to_string(),
                [-2.0, -2.0, -1.0, 0.0, 3.0, 4.0, 3.0, 1.0, 0.0, -1.0, -2.0],
            ),
        ]);

        let sample_rate = 44100;
        
        Self {
            running: false,
            eq_gains: [0.0; BANDS],
            input_device_index: 0,
            output_device_index: 0,
            devices: Vec::new(),
            presets,
            eq_processor: EQProcessor::new(sample_rate),
            sample_rate,
            spectrum_data: vec![0.0; SPECTRUM_POINTS],
            spectrum_peak: vec![0.0; SPECTRUM_POINTS],
            effects_chain: EffectsChain::new(sample_rate),
        }
    }
}
