use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, SampleRate, Stream};
use eframe::{App, CreationContext, egui};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::vec::Vec;

use crate::audio::devices::{
    create_stream_config, get_input_device, get_output_device, init_devices,
};
use crate::audio::{AudioState, SpectrumAnalyzer};
use crate::config::{BANDS, BUFFER_SIZE, CHANNELS, SPECTRUM_DECAY, SPECTRUM_POINTS};
use crate::effects::EffectsChain;
use crate::effects::eqprocessor::EQProcessor;
use crate::ui::{draw_effects_ui, draw_eq_sliders, draw_eq_visualization, draw_spectrum};

#[derive(PartialEq)]
enum Tab {
    Equalizer,
    Effects,
}

pub struct SystemWideEQ {
    audio_state: Arc<Mutex<AudioState>>,
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    host: Host,
    audio_buffer: Option<Arc<Mutex<Vec<f32>>>>,
    spectrum_analyzer: Arc<Mutex<SpectrumAnalyzer>>,
    current_tab: Tab,
}

impl SystemWideEQ {
    pub fn new(_cc: &CreationContext) -> Self {
        let host = cpal::default_host();
        let audio_state = Arc::new(Mutex::new(AudioState::default()));
        let spectrum_analyzer = Arc::new(Mutex::new(SpectrumAnalyzer::new()));

        let mut eq = Self {
            audio_state,
            input_stream: None,
            output_stream: None,
            host,
            audio_buffer: None,
            spectrum_analyzer,
            current_tab: Tab::Equalizer,
        };

        eq.init_devices();

        eq
    }

    fn init_devices(&mut self) {
        let (devices, input_device_index, output_device_index) = init_devices(&self.host);

        let mut state = self.audio_state.lock().unwrap();
        state.devices = devices;
        state.input_device_index = input_device_index;
        state.output_device_index = output_device_index;
    }

    pub fn start_processing(&mut self) -> Result<(), String> {
        {
            let state = self.audio_state.lock().unwrap();
            if state.running {
                return Ok(());
            }
        }

        let (input_device, output_device, sample_rate) = self.get_audio_devices()?;
        let actual_sample_rate = sample_rate.0;

        {
            let mut state = self.audio_state.lock().unwrap();
            state.sample_rate = actual_sample_rate;
            state
                .eq_processor_left
                .update_sample_rate(actual_sample_rate);
            state
                .eq_processor_right
                .update_sample_rate(actual_sample_rate);
            // state.effects_chain.update_sample_rate(actual_sample_rate);
        }

        let audio_buffer = self.setup_audio_buffer();

        let input_stream =
            self.create_input_stream(&input_device, sample_rate, audio_buffer.clone())?;
        let output_stream = self.create_output_stream(&output_device, sample_rate, audio_buffer)?;

        input_stream
            .play()
            .map_err(|e| format!("Failed to start input stream: {}", e))?;

        output_stream
            .play()
            .map_err(|e| format!("Failed to start output stream: {}", e))?;

        self.input_stream = Some(input_stream);
        self.output_stream = Some(output_stream);

        let mut state = self.audio_state.lock().unwrap();
        state.running = true;

        Ok(())
    }

    fn get_audio_devices(&self) -> Result<(Device, Device, SampleRate), String> {
        let state = self.audio_state.lock().unwrap();

        let input_device_index = state.input_device_index;
        let output_device_index = state.output_device_index;
        let devices = state.devices.clone();

        drop(state);

        let input_device = get_input_device(&self.host, &devices, input_device_index)
            .ok_or_else(|| "No input device selected".to_string())?;

        let output_device = get_output_device(&self.host, &devices, output_device_index)
            .ok_or_else(|| "No output device selected".to_string())?;

        let input_config = input_device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {}", e))?;

        if input_config.sample_format() != SampleFormat::F32 {
            return Err("Input device doesn't support F32 format".to_string());
        }

        let sample_rate = input_config.sample_rate();

        let output_supported_configs = output_device
            .supported_output_configs()
            .map_err(|e| format!("Failed to get output supported configs: {}", e))?;

        let supports_rate_and_format = output_supported_configs.into_iter().any(|conf| {
            conf.sample_format() == SampleFormat::F32
                && conf.min_sample_rate() <= sample_rate
                && conf.max_sample_rate() >= sample_rate
                && conf.channels() == CHANNELS as u16
        });

        if !supports_rate_and_format {
            eprintln!(
                "Warning: Output device might not directly support input sample rate/format/channels. Attempting anyway."
            );
        }

        Ok((input_device, output_device, sample_rate))
    }

    fn setup_audio_buffer(&mut self) -> Arc<Mutex<Vec<f32>>> {
        let buffer_size = BUFFER_SIZE * CHANNELS * 4;
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(buffer_size)));
        self.audio_buffer = Some(buffer.clone());
        buffer
    }

    fn create_input_stream(
        &self,
        input_device: &Device,
        sample_rate: SampleRate,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Stream, String> {
        let stream_config = create_stream_config(CHANNELS as u16, sample_rate);

        input_device
            .build_input_stream(
                &stream_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    process_input_audio(data, buffer.clone());
                },
                |err| eprintln!("Input stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build input stream: {}", e))
    }

    fn create_output_stream(
        &self,
        output_device: &Device,
        sample_rate: SampleRate,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Stream, String> {
        let audio_state_weak = Arc::downgrade(&self.audio_state);
        let spectrum_analyzer_weak = Arc::downgrade(&self.spectrum_analyzer);
        let stream_config = create_stream_config(CHANNELS as u16, sample_rate);

        let output_config = output_device
            .default_output_config()
            .map_err(|e| format!("Failed to get output config: {}", e))?;

        if output_config.sample_format() != SampleFormat::F32 {
            return Err(format!(
                "Output device default config is not F32, but {:?}. Check device support.",
                output_config.sample_format()
            ));
        }

        output_device
            .build_output_stream(
                &stream_config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    process_output_audio(
                        data,
                        buffer.clone(),
                        audio_state_weak.clone(),
                        spectrum_analyzer_weak.clone(),
                        sample_rate.0,
                    );
                },
                |err| eprintln!("Output stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {}", e))
    }

    pub fn stop_processing(&mut self) {
        self.input_stream = None;
        self.output_stream = None;
        self.audio_buffer = None;

        let mut state = self.audio_state.lock().unwrap();
        state.running = false;
    }

    pub fn load_preset(&mut self, preset_name: &str) {
        let mut state = self.audio_state.lock().unwrap();
        if let Some(preset_gains) = state.presets.get(preset_name).cloned() {
            state.eq_gains = preset_gains;
            state.eq_processor_left.update_gains(preset_gains);
            state.eq_processor_right.update_gains(preset_gains);
        }
    }
}

fn process_input_audio(data: &[f32], buffer: Arc<Mutex<Vec<f32>>>) {
    if let Ok(mut buffer_lock) = buffer.lock() {
        buffer_lock.extend_from_slice(data);

        let max_buffer_len = BUFFER_SIZE * CHANNELS * 10;
        if buffer_lock.len() > max_buffer_len {
            let excess = buffer_lock.len() - max_buffer_len;
            buffer_lock.drain(0..excess);
        }
    }
}

fn update_spectrum_data(analyzer: &mut SpectrumAnalyzer, state: &mut AudioState) {
    let sample_rate = state.sample_rate;
    let spectrum = analyzer.process_fft(sample_rate);

    let len_to_process = SPECTRUM_POINTS.min(spectrum.len());

    for i in 0..len_to_process {
        state.spectrum_data[i] =
            state.spectrum_data[i] * SPECTRUM_DECAY + spectrum[i] * (1.0 - SPECTRUM_DECAY);

        if state.spectrum_data[i] > state.spectrum_peak[i] {
            state.spectrum_peak[i] = state.spectrum_data[i];
        } else {
            state.spectrum_peak[i] *= 0.98;
        }
    }
    for i in len_to_process..SPECTRUM_POINTS {
        state.spectrum_data[i] = state.spectrum_data[i] * SPECTRUM_DECAY;
        state.spectrum_peak[i] *= 0.98;
    }
}

fn process_output_audio(
    data: &mut [f32],
    buffer: Arc<Mutex<Vec<f32>>>,
    audio_state_weak: Weak<Mutex<AudioState>>,
    spectrum_analyzer_weak: Weak<Mutex<SpectrumAnalyzer>>,
    _sample_rate: u32,
) {
    let samples_needed = data.len();
    let mut input_samples = Vec::with_capacity(samples_needed);

    if let Ok(mut buffer_lock) = buffer.lock() {
        let available = buffer_lock.len();
        let samples_to_take = available.min(samples_needed);

        if samples_to_take > 0 {
            input_samples.extend_from_slice(&buffer_lock[0..samples_to_take]);
            buffer_lock.drain(0..samples_to_take);
        }
        if samples_to_take < samples_needed {
            input_samples.resize(samples_needed, 0.0);
        }
    } else {
        input_samples.resize(samples_needed, 0.0);
    }

    if let Some(state_arc) = audio_state_weak.upgrade() {
        if let Ok(mut state) = state_arc.lock() {
            let num_frames = samples_needed / CHANNELS;
            for i in 0..num_frames {
                let in_l_idx = i * CHANNELS;
                let out_l_idx = i * CHANNELS;

                let processed_l = state
                    .eq_processor_left
                    .process_sample(input_samples[in_l_idx]);
                data[out_l_idx] = processed_l;

                if CHANNELS == 2 {
                    let in_r_idx = i * CHANNELS + 1;
                    let out_r_idx = i * CHANNELS + 1;

                    let processed_r = state
                        .eq_processor_right
                        .process_sample(input_samples[in_r_idx]);
                    data[out_r_idx] = processed_r;
                }
            }
        } else {
            data.copy_from_slice(&input_samples);
        }
    } else {
        data.copy_from_slice(&input_samples);
    }

    // if let Some(state_arc) = audio_state_weak.upgrade() {
    //     if let Ok(mut state) = state_arc.lock() {
    //         state.effects_chain.process(data, state.sample_rate);
    //     }
    // }

    if let Some(analyzer_arc) = spectrum_analyzer_weak.upgrade() {
        if let Ok(mut analyzer) = analyzer_arc.lock() {
            if analyzer.add_samples(data) {
                if let Some(state_arc) = audio_state_weak.upgrade() {
                    if let Ok(mut state) = state_arc.lock() {
                        update_spectrum_data(&mut analyzer, &mut state);
                    }
                }
            }
        }
    }
}

impl App for SystemWideEQ {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let running = {
            let state = self.audio_state.lock().unwrap();
            state.running
        };

        if running {
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Exit").clicked() {
                        self.stop_processing();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Presets", |ui| {
                    let preset_names: Vec<String> = {
                        let state = self.audio_state.lock().unwrap();
                        state.presets.keys().cloned().collect()
                    };

                    for preset_name in preset_names {
                        if ui.button(&preset_name).clicked() {
                            self.load_preset(&preset_name);
                            ui.close_menu();
                        }
                    }
                });

                if ui
                    .button(if running {
                        "Stop Processing"
                    } else {
                        "Start Processing"
                    })
                    .clicked()
                {
                    if running {
                        self.stop_processing();
                    } else {
                        if let Err(e) = self.start_processing() {
                            eprintln!("Failed to start processing: {}", e);
                        }
                    }
                }

                if ui.button("Reset EQ").clicked() {
                    let mut state = self.audio_state.lock().unwrap();
                    let zero_gains = [0.0; BANDS];
                    state.eq_gains = zero_gains;
                    state.eq_processor_left.update_gains(zero_gains);
                    state.eq_processor_right.update_gains(zero_gains);
                }

                let status_text = if running {
                    "Status: Running"
                } else {
                    "Status: Stopped"
                };
                let status_color = if running {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                ui.label(egui::RichText::new(status_text).color(status_color));

                ui.separator();
                ui.selectable_value(&mut self.current_tab, Tab::Equalizer, "Equalizer");
                ui.selectable_value(&mut self.current_tab, Tab::Effects, "Effects");
            });
        });

        egui::SidePanel::right("devices_panel").show(ctx, |ui| {
            ui.heading("Audio Devices");
            ui.separator();

            let (devices, mut input_idx, mut output_idx) = {
                let state = self.audio_state.lock().unwrap();
                (
                    state.devices.clone(),
                    state.input_device_index,
                    state.output_device_index,
                )
            };

            let mut current_input_idx = input_idx;
            let mut current_output_idx = output_idx;
            let mut input_changed = false;
            let mut output_changed = false;

            egui::ComboBox::from_label("Input Device")
                .selected_text(
                    devices
                        .get(current_input_idx)
                        .map(|s| s.trim_start_matches("Input: "))
                        .unwrap_or("None Selected"),
                )
                .show_ui(ui, |ui| {
                    for (i, device_name) in devices.iter().enumerate() {
                        if device_name.starts_with("Input:") {
                            let text = device_name.trim_start_matches("Input: ");
                            if ui
                                .selectable_value(&mut current_input_idx, i, text)
                                .changed()
                            {
                                input_changed = true;
                            }
                        }
                    }
                });

            if input_changed {
                input_idx = current_input_idx;
            }

            egui::ComboBox::from_label("Output Device")
                .selected_text(
                    devices
                        .get(current_output_idx)
                        .map(|s| s.trim_start_matches("Output: "))
                        .unwrap_or("None Selected"),
                )
                .show_ui(ui, |ui| {
                    for (i, device_name) in devices.iter().enumerate() {
                        if device_name.starts_with("Output:") {
                            let text = device_name.trim_start_matches("Output: ");
                            if ui
                                .selectable_value(&mut current_output_idx, i, text)
                                .changed()
                            {
                                output_changed = true;
                            }
                        }
                    }
                });

            if output_changed {
                output_idx = current_output_idx;
            }

            if input_changed || output_changed {
                let mut state = self.audio_state.lock().unwrap();
                state.input_device_index = input_idx;
                state.output_device_index = output_idx;
            }

            if ui.button("Apply Device Settings").clicked() {
                let was_running = running;
                if was_running {
                    self.stop_processing();
                }
                if let Err(e) = self.start_processing() {
                    eprintln!(
                        "Failed to start/restart processing after device change: {}",
                        e
                    );
                }
            }

            if ui.button("Refresh Device List").clicked() {
                let (new_devices, new_input_idx, new_output_idx) = init_devices(&self.host);
                let mut state = self.audio_state.lock().unwrap();
                state.devices = new_devices;
                state.input_device_index = new_input_idx.min(state.devices.len().saturating_sub(1));
                state.output_device_index =
                    new_output_idx.min(state.devices.len().saturating_sub(1));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            Tab::Equalizer => {
                ui.heading("Equalizer");
                ui.separator();

                let (spectrum_data, spectrum_peak) = {
                    let state = self.audio_state.lock().unwrap();
                    (state.spectrum_data, state.spectrum_peak)
                };
                draw_spectrum(ui, &spectrum_data, &spectrum_peak);

                let mut current_eq_gains = {
                    let state = self.audio_state.lock().unwrap();
                    state.eq_gains
                };
                draw_eq_visualization(ui, &current_eq_gains);

                ui.separator();

                if draw_eq_sliders(ui, &mut current_eq_gains) {
                    let mut state = self.audio_state.lock().unwrap();
                    state.eq_gains = current_eq_gains;
                    let gains = state.eq_gains;
                    state.eq_processor_left.update_gains(gains);
                    state.eq_processor_right.update_gains(gains);
                }
            }
            Tab::Effects => {
                ui.heading("Audio Effects");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut state = self.audio_state.lock().unwrap();
                    draw_effects_ui(ui, &mut state.effects_chain);

                    ui.separator();
                    if ui.button("Reset All Effects").clicked() {
                        let mut state = self.audio_state.lock().unwrap();
                        state.effects_chain = EffectsChain::new(state.sample_rate);
                    }
                });
            }
        });
    }
}
