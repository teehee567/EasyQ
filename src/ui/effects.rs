use eframe::egui::{self, RichText, Ui};
use crate::effects::{
    compressor::CompressorEffect, delay::DelayEffect, distortion::DistortionEffect, reverb::ReverbEffect, AudioEffect, EffectsChain
};

fn draw_effect_header(ui: &mut Ui, name: &str, enabled: &mut bool) -> bool {
    ui.horizontal(|ui| {
        let title = RichText::new(name).size(18.0).strong();
        ui.heading(title);
        ui.add_space(8.0);
        ui.checkbox(enabled, "Enabled").changed()
    }).inner
}

fn param_slider<T>(ui: &mut Ui, value: &mut T, range: std::ops::RangeInclusive<T>, text: &str) -> bool
where
    T: egui::emath::Numeric,
{
    ui.horizontal(|ui| {
        ui.label(text);
        ui.add(egui::Slider::new(value, range).text(""))
    }).inner.changed()
}

pub fn draw_reverb_ui(ui: &mut egui::Ui, reverb: &mut ReverbEffect) -> bool {
    let mut changed = false;
    
    ui.add_space(4.0);
    egui::Frame::new()
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            
            let mut enabled = reverb.is_enabled();
            if draw_effect_header(ui, "REVERB", &mut enabled) {
                reverb.set_enabled(enabled);
                changed = true;
            }
            
            ui.separator();
            
            if enabled {
                ui.columns(2, |columns| {
                    // Left column
                    let mut room_size = reverb.room_size;
                    if param_slider(&mut columns[0], &mut room_size, 0.0..=1.0, "Room Size") {
                        reverb.set_room_size(room_size);
                        changed = true;
                    }
                    
                    let mut damping = reverb.damping;
                    if param_slider(&mut columns[0], &mut damping, 0.0..=1.0, "Damping") {
                        reverb.set_damping(damping);
                        changed = true;
                    }
                    
                    // Right column
                    let mut wet_level = reverb.wet_level;
                    if param_slider(&mut columns[1], &mut wet_level, 0.0..=1.0, "Wet Level") {
                        reverb.set_wet_level(wet_level);
                        changed = true;
                    }
                    
                    let mut dry_level = reverb.dry_level;
                    if param_slider(&mut columns[1], &mut dry_level, 0.0..=1.0, "Dry Level") {
                        reverb.set_dry_level(dry_level);
                        changed = true;
                    }
                });
                
                let mut freeze_mode = reverb.freeze_mode;
                if ui.checkbox(&mut freeze_mode, "Freeze Mode").changed() {
                    reverb.set_freeze_mode(freeze_mode);
                    changed = true;
                }
            }
        });
    
    ui.add_space(8.0);
    changed
}

pub fn draw_delay_ui(ui: &mut egui::Ui, delay: &mut DelayEffect) -> bool {
    let mut changed = false;
    
    ui.add_space(4.0);
    egui::Frame::new()
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            
            let mut enabled = delay.is_enabled();
            if draw_effect_header(ui, "DELAY", &mut enabled) {
                delay.set_enabled(enabled);
                changed = true;
            }
            
            ui.separator();
            
            if enabled {
                let mut delay_time = delay.delay_time_ms;
                if param_slider(ui, &mut delay_time, 10.0..=2000.0, "Delay Time (ms)") {
                    delay.set_delay_time(delay_time);
                    changed = true;
                }
                
                ui.columns(2, |columns| {
                    // Left column
                    let mut feedback = delay.feedback;
                    if param_slider(&mut columns[0], &mut feedback, 0.0..=0.95, "Feedback") {
                        delay.set_feedback(feedback);
                        changed = true;
                    }
                    
                    // Right column
                    let mut mix = delay.mix;
                    if param_slider(&mut columns[1], &mut mix, 0.0..=1.0, "Mix") {
                        delay.set_mix(mix);
                        changed = true;
                    }
                });
            }
        });
    
    ui.add_space(8.0);
    changed
}

pub fn draw_distortion_ui(ui: &mut egui::Ui, distortion: &mut DistortionEffect) -> bool {
    let mut changed = false;
    
    ui.add_space(4.0);
    egui::Frame::new()
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            
            let mut enabled = distortion.is_enabled();
            if draw_effect_header(ui, "DISTORTION", &mut enabled) {
                distortion.set_enabled(enabled);
                changed = true;
            }
            
            ui.separator();
            
            if enabled {
                ui.columns(2, |columns| {
                    // Left column
                    let mut drive = distortion.drive;
                    if param_slider(&mut columns[0], &mut drive, 1.0..=100.0, "Drive") {
                        distortion.set_drive(drive);
                        changed = true;
                    }
                    
                    let mut tone = distortion.tone;
                    if param_slider(&mut columns[0], &mut tone, 0.0..=1.0, "Tone") {
                        distortion.set_tone(tone);
                        changed = true;
                    }
                    
                    // Right column
                    let mut mix = distortion.mix;
                    if param_slider(&mut columns[1], &mut mix, 0.0..=1.0, "Mix") {
                        distortion.set_mix(mix);
                        changed = true;
                    }
                    
                    let mut output_gain = distortion.output_gain;
                    if param_slider(&mut columns[1], &mut output_gain, 0.0..=1.0, "Output Gain") {
                        distortion.set_output_gain(output_gain);
                        changed = true;
                    }
                });
            }
        });
    
    ui.add_space(8.0);
    changed
}

pub fn draw_compressor_ui(ui: &mut egui::Ui, compressor: &mut CompressorEffect) -> bool {
    let mut changed = false;
    
    ui.add_space(4.0);
    egui::Frame::new()
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            
            let mut enabled = compressor.is_enabled();
            if draw_effect_header(ui, "COMPRESSOR", &mut enabled) {
                compressor.set_enabled(enabled);
                changed = true;
            }
            
            ui.separator();
            
            if enabled {
                ui.columns(2, |columns| {
                    let mut threshold = compressor.threshold;
                    if param_slider(&mut columns[0], &mut threshold, -60.0..=0.0, "Threshold (dB)") {
                        compressor.set_threshold(threshold);
                        changed = true;
                    }
                    
                    let mut ratio = compressor.ratio;
                    if param_slider(&mut columns[0], &mut ratio, 1.0..=20.0, "Ratio") {
                        compressor.set_ratio(ratio);
                        changed = true;
                    }
                    
                    let mut makeup_gain = compressor.makeup_gain;
                    if param_slider(&mut columns[0], &mut makeup_gain, 0.0..=24.0, "Makeup Gain (dB)") {
                        compressor.set_makeup_gain(makeup_gain);
                        changed = true;
                    }
                    
                    // Right column
                    let mut attack = compressor.attack;
                    if param_slider(&mut columns[1], &mut attack, 0.1..=200.0, "Attack (ms)") {
                        compressor.set_attack(attack);
                        changed = true;
                    }
                    
                    let mut release = compressor.release;
                    if param_slider(&mut columns[1], &mut release, 10.0..=2000.0, "Release (ms)") {
                        compressor.set_release(release);
                        changed = true;
                    }
                });
            }
        });
    
    ui.add_space(8.0);
    changed
}

pub fn draw_effects_ui(
    ui: &mut egui::Ui, 
    effects_chain: &mut EffectsChain,
) -> bool {
    let mut changed = false;
    
    ui.vertical(|ui| {
        ui.add_space(4.0);
        ui.heading(RichText::new("Audio Effects").size(22.0).strong());
        ui.add_space(8.0);
        
        ui.vertical(|ui| {
            ui.set_min_width(280.0);
            changed |= draw_distortion_ui(ui, effects_chain.distortion());
        });

        ui.vertical(|ui| {
            ui.set_min_width(280.0);
            changed |= draw_delay_ui(ui, effects_chain.delay());
        });
        
        ui.vertical(|ui| {
            ui.set_min_width(280.0);
            changed |= draw_compressor_ui(ui, effects_chain.compressor());
        });
        
        
        ui.vertical(|ui| {
            ui.set_min_width(280.0);
            changed |= draw_reverb_ui(ui, effects_chain.reverb());
        });
    });
    
    changed
}
