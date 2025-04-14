pub mod reverb;
pub mod delay;
pub mod distortion;
pub mod compressor;
pub mod eqprocessor;

pub trait AudioEffect {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32);
    fn reset(&mut self);
    fn is_enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
    fn name(&self) -> &str;
}

pub struct EffectsChain {
    reverb: reverb::ReverbEffect,
    delay: delay::DelayEffect,
    distortion: distortion::DistortionEffect,
    compressor: compressor::CompressorEffect,
}

impl EffectsChain {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            reverb: reverb::ReverbEffect::new(sample_rate),
            delay: delay::DelayEffect::new(sample_rate),
            distortion: distortion::DistortionEffect::new(),
            compressor: compressor::CompressorEffect::new(),
        }
    }
    
    pub fn process(&mut self, buffer: &mut [f32], sample_rate: u32) {
        self.distortion.process(buffer, sample_rate);
        self.compressor.process(buffer, sample_rate);
        self.delay.process(buffer, sample_rate);
        self.reverb.process(buffer, sample_rate);
    }
    
    pub fn reverb(&mut self) -> &mut reverb::ReverbEffect {
        &mut self.reverb
    }
    
    pub fn delay(&mut self) -> &mut delay::DelayEffect {
        &mut self.delay
    }
    
    pub fn distortion(&mut self) -> &mut distortion::DistortionEffect {
        &mut self.distortion
    }

    pub fn compressor(&mut self) -> &mut compressor::CompressorEffect {
        &mut self.compressor
    }
}
