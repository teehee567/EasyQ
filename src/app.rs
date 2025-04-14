use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Host, SampleFormat, Stream};
use eframe::{App, CreationContext, egui};
use std::sync::{Arc, Mutex, Weak};

use crate::audio::devices::{
    create_stream_config, get_input_device, get_output_device, init_devices,
};
use crate::audio::{AudioState, SpectrumAnalyzer};
use crate::config::{BANDS, BUFFER_SIZE, CHANNELS, SPECTRUM_DECAY, SPECTRUM_POINTS};
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
        // Initialize audio host
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

    // Initialize audio devices
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

        {
            let mut state = self.audio_state.lock().unwrap();
            state.sample_rate = sample_rate.0;
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

    fn get_audio_devices(&self) -> Result<(cpal::Device, cpal::Device, cpal::SampleRate), String> {
        let state = self.audio_state.lock().unwrap();

        let input_device_index = state.input_device_index;
        let output_device_index = state.output_device_index;
        let devices = state.devices.clone();

        drop(state);

        // Get selected devices
        let input_device = get_input_device(&self.host, &devices, input_device_index)
            .ok_or_else(|| "No input device selected".to_string())?;

        let output_device = get_output_device(&self.host, &devices, output_device_index)
            .ok_or_else(|| "No output device selected".to_string())?;

        // Get default configs for devices
        let input_config = input_device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {}", e))?;

        if input_config.sample_format() != SampleFormat::F32 {
            return Err("Input device doesn't support F32 format".to_string());
        }

        let sample_rate = input_config.sample_rate();

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
        input_device: &cpal::Device,
        sample_rate: cpal::SampleRate,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream, String> {
        let stream_config = create_stream_config(CHANNELS as u16, sample_rate);

        input_device
            .build_input_stream(
                &stream_config,
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
        output_device: &cpal::Device,
        sample_rate: cpal::SampleRate,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream, String> {
        let audio_state_weak = Arc::downgrade(&self.audio_state);
        let spectrum_analyzer_weak = Arc::downgrade(&self.spectrum_analyzer);
        let stream_config = create_stream_config(CHANNELS as u16, sample_rate);

        let output_config = output_device
            .default_output_config()
            .map_err(|e| format!("Failed to get output config: {}", e))?;

        if output_config.sample_format() != SampleFormat::F32 {
            return Err("Output device doesn't support F32 format".to_string());
        }

        output_device
            .build_output_stream(
                &stream_config,
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
        let eq_gains = state.eq_gains;

        if let Some(preset) = state.presets.get(preset_name) {
            state.eq_gains = *preset;
            state.eq_processor.update_gains(eq_gains);
        }
    }
}

fn process_input_audio(data: &[f32], buffer: Arc<Mutex<Vec<f32>>>) {
    if let Ok(mut buffer) = buffer.lock() {
        if CHANNELS == 2 {
            for i in (0..data.len()).step_by(2) {
                if i < data.len() {
                    buffer.push(data[i]);
                    buffer.push(data[i]);
                }
            }
        } else {
            buffer.extend_from_slice(data);
        }


        // Keep your buffer size management
        let buffer_size = BUFFER_SIZE * CHANNELS * 4;
        if buffer.len() > buffer_size {
            let excess = buffer.len() - buffer_size;
            buffer.drain(0..excess);
        }
    }
}

fn update_spectrum_data(analyzer: &mut SpectrumAnalyzer, state: &mut AudioState) {
    let sample_rate = state.sample_rate;
    let spectrum = analyzer.process_fft(sample_rate);

    // Apply temporal smoothing
    for i in 0..SPECTRUM_POINTS {
        // Smooth current data
        state.spectrum_data[i] =
            state.spectrum_data[i] * SPECTRUM_DECAY + spectrum[i] * (1.0 - SPECTRUM_DECAY);

        if state.spectrum_data[i] > state.spectrum_peak[i] {
            state.spectrum_peak[i] = state.spectrum_data[i];
        } else {
            state.spectrum_peak[i] *= 0.98;
        }
    }
}

fn process_output_audio(
    data: &mut [f32],
    buffer: Arc<Mutex<Vec<f32>>>,
    audio_state_weak: Weak<Mutex<AudioState>>,
    spectrum_analyzer_weak: Weak<Mutex<SpectrumAnalyzer>>,
    sample_rate: u32,
) {
    // Update EQ processor
    if let Some(state) = audio_state_weak.upgrade() {
        if let Ok(mut state) = state.lock() {
            let eq_gains = state.eq_gains;
            state.eq_processor.update_sample_rate(sample_rate);
            state.eq_processor.update_gains(eq_gains);
        }
    }

    // Get input buffer
    let input_samples = if let Ok(mut buffer_lock) = buffer.lock() {
        // Take all samples from the buffer
        let mut samples = Vec::new();
        std::mem::swap(&mut samples, &mut buffer_lock);
        samples
    } else {
        Vec::new()
    };


// Print a message if the input buffer has any non-zero right channel samples
if CHANNELS == 2 && !input_samples.is_empty() {
    let mut right_channel_has_sound = false;
    for i in (1..input_samples.len()).step_by(2) {
        if input_samples[i].abs() > 0.01 {
            right_channel_has_sound = true;
            break;
        }
    }
    
    if right_channel_has_sound {
        println!("Right channel input detected");
    } else {
        println!("No right channel input detected");
    }
}

    if CHANNELS == 2 {
        // Stereo processing
        let output_frames = data.len() / 2;
        
        for frame in 0..output_frames {
            let left_out_idx = frame * 2;
            let right_out_idx = frame * 2 + 1;
            
            // Get input samples - already should be in stereo format due to process_input_audio
            let (left_in, right_in) = if frame * 2 + 1 < input_samples.len() {
                (input_samples[frame * 2], input_samples[frame * 2 + 1])
            } else {
                (0.0, 0.0)
            };
            
            // Process each channel
            let left_processed = apply_eq_processing(left_in, &audio_state_weak);
            let right_processed = apply_eq_processing(right_in, &audio_state_weak);
            
            // Set output data
            if left_out_idx < data.len() {
                data[left_out_idx] = left_processed;
            }
            
            if right_out_idx < data.len() {
                data[right_out_idx] = right_processed;
            }
        }
    } else {
        // Mono processing
        let output_frames = data.len();
        for i in 0..output_frames {
            let in_sample = if i < input_samples.len() { input_samples[i] } else { 0.0 };
            let processed = apply_eq_processing(in_sample, &audio_state_weak);
            data[i] = processed;
        }
    }

    // Apply effects chain
    apply_effects_chain(data, sample_rate, audio_state_weak.clone());

    // Spectrum analysis
    if let Some(analyzer) = spectrum_analyzer_weak.upgrade() {
        if let Ok(mut analyzer) = analyzer.lock() {
            if analyzer.add_samples(data) {
                if let Some(state) = audio_state_weak.upgrade() {
                    if let Ok(mut state) = state.lock() {
                        update_spectrum_data(&mut analyzer, &mut state);
                    }
                }
            }
        }
    }
}

fn apply_eq_processing(sample: f32, audio_state_weak: &Weak<Mutex<AudioState>>) -> f32 {
    if let Some(state) = audio_state_weak.upgrade() {
        if let Ok(mut state) = state.lock() {
            return state.eq_processor.process(sample);
        }
    }

    sample
}

fn apply_effects_chain(
    data: &mut [f32],
    sample_rate: u32,
    audio_state_weak: Weak<Mutex<AudioState>>,
) {
    if let Some(state) = audio_state_weak.upgrade() {
        if let Ok(mut state) = state.lock() {
            state.effects_chain.process(data, sample_rate);
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

        // controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Exit").clicked() {
                        std::process::exit(0);
                    }
                });

                ui.menu_button("Presets", |ui| {
                    // Clone the preset names to avoid borrow issues
                    let preset_names: Vec<String> = {
                        let state = self.audio_state.lock().unwrap();
                        state.presets.keys().cloned().collect()
                    };

                    for preset_name in preset_names {
                        if ui.button(&preset_name).clicked() {
                            self.load_preset(&preset_name);
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
                    state.eq_gains = [0.0; BANDS];
                }

                ui.label(if running {
                    "Status: Running"
                } else {
                    "Status: Stopped"
                });

                // Add tab buttons
                ui.separator();
                ui.selectable_value(&mut self.current_tab, Tab::Equalizer, "Equalizer");
                ui.selectable_value(&mut self.current_tab, Tab::Effects, "Effects");
            });
        });

        // device selection
        egui::SidePanel::right("devices_panel").show(ctx, |ui| {
            ui.heading("Audio Devices");

            let (devices, input_idx, output_idx) = {
                let state = self.audio_state.lock().unwrap();
                (
                    state.devices.clone(),
                    state.input_device_index,
                    state.output_device_index,
                )
            };

            let mut new_input_idx = input_idx;
            let mut new_output_idx = output_idx;
            let mut device_changed = false;

            egui::ComboBox::from_label("Input Device")
                .selected_text(
                    devices
                        .get(input_idx)
                        .map(|s| s.trim_start_matches("Input: "))
                        .unwrap_or("None"),
                )
                .show_ui(ui, |ui| {
                    for (i, device_name) in devices.iter().enumerate() {
                        if device_name.starts_with("Input:") {
                            if ui
                                .selectable_value(
                                    &mut new_input_idx,
                                    i,
                                    device_name.trim_start_matches("Input: "),
                                )
                                .clicked()
                            {
                                device_changed = true;
                            }
                        }
                    }
                });

            egui::ComboBox::from_label("Output Device")
                .selected_text(
                    devices
                        .get(output_idx)
                        .map(|s| s.trim_start_matches("Output: "))
                        .unwrap_or("None"),
                )
                .show_ui(ui, |ui| {
                    for (i, device_name) in devices.iter().enumerate() {
                        if device_name.starts_with("Output:") {
                            if ui
                                .selectable_value(
                                    &mut new_output_idx,
                                    i,
                                    device_name.trim_start_matches("Output: "),
                                )
                                .clicked()
                            {
                                device_changed = true;
                            }
                        }
                    }
                });

            if device_changed {
                let mut state = self.audio_state.lock().unwrap();
                state.input_device_index = new_input_idx;
                state.output_device_index = new_output_idx;
            }

            if ui.button("Apply Device Settings").clicked() {
                if running {
                    self.stop_processing();
                    if let Err(e) = self.start_processing() {
                        eprintln!("Failed to restart processing: {}", e);
                    }
                }
            }
        });

        // main eq area
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_tab {
                Tab::Equalizer => {
                    ui.heading("Equalizer");

                    // Spectrum Analyzer
                    let (spectrum_data, spectrum_peak) = {
                        let state = self.audio_state.lock().unwrap();
                        (
                            state.spectrum_data.clone(),
                            state.spectrum_peak.clone(),
                        )
                    };
                    draw_spectrum(ui, &spectrum_data, &spectrum_peak);

                    // Eq visualization
                    let eq_gains = {
                        let state = self.audio_state.lock().unwrap();
                        state.eq_gains
                    };
                    draw_eq_visualization(ui, &eq_gains);

                    // Eq Slider
                    let mut eq_gains = {
                        let state = self.audio_state.lock().unwrap();
                        state.eq_gains
                    };

                    if draw_eq_sliders(ui, &mut eq_gains) {
                        let mut state = self.audio_state.lock().unwrap();
                        state.eq_gains = eq_gains;
                    }
                }
                Tab::Effects => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut state = self.audio_state.lock().unwrap();

                        draw_effects_ui(ui, &mut state.effects_chain);

                        if ui.button("Reset All Effects").clicked() {
                            let mut state = self.audio_state.lock().unwrap();
                            state.effects_chain =
                                crate::effects::EffectsChain::new(state.sample_rate);
                        }
                    });
                }
            }
        });
    }
}
