use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, Host, SampleFormat, Stream, StreamConfig};
use eframe::{App, CreationContext, NativeOptions, egui};
use ringbuf::HeapRb;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rustfft::{FftPlanner, num_complex::{Complex, Complex32}};
use apodize::hanning_iter;

// EQ settings
const BANDS: usize = 10;
const FREQUENCIES: [f32; BANDS] = [
    31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];
const BUFFER_SIZE: usize = 1024;
const CHANNELS: usize = 2;

// Spectrum analyzer settings
const SPECTRUM_POINTS: usize = 512; // Number of FFT points to display
const SPECTRUM_HISTORY: usize = 3;  // Number of frames to average for smoother display
const SPECTRUM_DECAY: f32 = 0.7;    // Decay factor for visualization smoothing

// FFT settings
const FFT_SIZE: usize = 2048;
const OVERLAP: usize = FFT_SIZE / 2;

// Structure for audio processing status and controls
struct AudioState {
    running: bool,
    eq_gains: [f32; BANDS],
    input_device_index: usize,
    output_device_index: usize,
    devices: Vec<String>,
    presets: HashMap<String, [f32; BANDS]>,
    sample_rate: u32,
    spectrum_data: Vec<f32>,        // Store the current spectrum data
    spectrum_peak: Vec<f32>,        // Store peak values for persistence
}

fn lerp_rgba(from: egui::Rgba, to: egui::Rgba, t: f32) -> egui::Rgba {
    egui::Rgba::from_rgba_premultiplied(
        from.r() + (to.r() - from.r()) * t,
        from.g() + (to.g() - from.g()) * t,
        from.b() + (to.b() - from.b()) * t,
        from.a() + (to.a() - from.a()) * t,
    )
}

impl Default for AudioState {
    fn default() -> Self {
        let presets = HashMap::from([
            ("Flat".to_string(), [0.0; BANDS]),
            (
                "Bass Boost".to_string(),
                [7.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "Treble Boost".to_string(),
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0],
            ),
            (
                "V-Shape".to_string(),
                [5.0, 3.0, 1.0, -1.0, -2.0, -2.0, -1.0, 1.0, 3.0, 5.0],
            ),
            (
                "Vocal Boost".to_string(),
                [-2.0, -2.0, -1.0, 0.0, 3.0, 4.0, 3.0, 1.0, 0.0, -1.0],
            ),
        ]);

        Self {
            running: false,
            eq_gains: [0.0; BANDS],
            input_device_index: 0,
            output_device_index: 0,
            devices: Vec::new(),
            presets,
            sample_rate: 44100,
            spectrum_data: vec![0.0; SPECTRUM_POINTS],
            spectrum_peak: vec![0.0; SPECTRUM_POINTS],
        }
    }
}

// Spectrum analyzer data
struct SpectrumAnalyzer {
    fft_planner: FftPlanner<f32>,
    fft_input: Vec<Complex32>,
    fft_output: Vec<Complex32>,
    window: Vec<f32>,
    buffer: Vec<f32>,
    buffer_pos: usize,
}

impl SpectrumAnalyzer {
    fn new() -> Self {
        let mut analyzer = Self {
            fft_planner: FftPlanner::new(),
            fft_input: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            fft_output: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window: hanning_iter(FFT_SIZE).map(|x| x as f32).collect(),
            buffer: vec![0.0; FFT_SIZE],
            buffer_pos: 0,
        };
        
        analyzer
    }

    fn add_samples(&mut self, samples: &[f32]) -> bool {
        let mut fft_ready = false;
        
        // Mix stereo to mono if needed and add to buffer
        for chunk in samples.chunks(CHANNELS) {
            if chunk.len() == CHANNELS {
                // Average all channels to mono
                let sample = chunk.iter().sum::<f32>() / (CHANNELS as f32);
                
                self.buffer[self.buffer_pos] = sample;
                self.buffer_pos += 1;
                
                if self.buffer_pos >= self.buffer.len() {
                    self.buffer_pos = 0;
                    fft_ready = true;
                }
            }
        }
        
        fft_ready
    }
    
    // Process FFT and return spectrum data
    fn process_fft(&mut self, sample_rate: u32) -> Vec<f32> {
        // Apply window function and prepare FFT input
        for i in 0..FFT_SIZE {
            self.fft_input[i] = Complex32::new(self.buffer[i] * self.window[i], 0.0);
        }
        
        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        self.fft_output.copy_from_slice(&self.fft_input);
        fft.process(&mut self.fft_output);
        
        // Calculate magnitude spectrum
        let mut spectrum = vec![0.0; SPECTRUM_POINTS];
        let scale_factor = 2.0 / (FFT_SIZE as f32);
        
        for i in 0..SPECTRUM_POINTS {
            // Calculate magnitude with proper scaling
            let mag = self.fft_output[i].norm() * scale_factor;
            
            // Convert to dB scale with improved range
            let db = 20.0 * (mag + 1e-10).log10();
            
            // Normalize to 0-60 range with floor at -80dB
            spectrum[i] = (db + 80.0).max(0.0).min(60.0);
        }
        
        spectrum
    }
}

// Main application structure
struct SystemWideEQ {
    audio_state: Arc<Mutex<AudioState>>,
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    host: Host,
    // Using a shared ring buffer
    audio_buffer: Option<Arc<Mutex<Vec<f32>>>>,
    spectrum_analyzer: Arc<Mutex<SpectrumAnalyzer>>,
}

impl SystemWideEQ {
    fn new(_cc: &CreationContext) -> Self {
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
        };

        eq.init_devices();

        eq
    }

    // Initialize audio devices
    fn init_devices(&mut self) {
        let mut state = self.audio_state.lock().unwrap();

        // Get available devices
        state.devices.clear();

        if let Ok(input_devices) = self.host.input_devices() {
            for device in input_devices {
                if let Ok(name) = device.name() {
                    state.devices.push(format!("Input: {}", name));
                }
            }
        }

        if let Ok(output_devices) = self.host.output_devices() {
            for device in output_devices {
                if let Ok(name) = device.name() {
                    state.devices.push(format!("Output: {}", name));
                }
            }
        }

        // Set default devices
        if let Some(default_input) = self.host.default_input_device() {
            if let Ok(name) = default_input.name() {
                for (i, device_name) in state.devices.iter().enumerate() {
                    if device_name.contains(&name) && device_name.starts_with("Input:") {
                        state.input_device_index = i;
                        break;
                    }
                }
            }
        }

        if let Some(default_output) = self.host.default_output_device() {
            if let Ok(name) = default_output.name() {
                for (i, device_name) in state.devices.iter().enumerate() {
                    if device_name.contains(&name) && device_name.starts_with("Output:") {
                        state.output_device_index = i;
                        break;
                    }
                }
            }
        }
    }

    // Get input device by index
    fn get_input_device(&self, index: usize) -> Option<Device> {
        let state = self.audio_state.lock().unwrap();

        if index >= state.devices.len() {
            return None;
        }

        let device_name = &state.devices[index];
        if !device_name.starts_with("Input:") {
            return None;
        }

        let device_name = device_name.trim_start_matches("Input: ");

        if let Ok(input_devices) = self.host.input_devices() {
            for device in input_devices {
                if let Ok(name) = device.name() {
                    if name == device_name {
                        return Some(device);
                    }
                }
            }
        }

        None
    }

    // Get output device by index
    fn get_output_device(&self, index: usize) -> Option<Device> {
        let state = self.audio_state.lock().unwrap();

        if index >= state.devices.len() {
            return None;
        }

        let device_name = &state.devices[index];
        if !device_name.starts_with("Output:") {
            return None;
        }

        let device_name = device_name.trim_start_matches("Output: ");

        if let Ok(output_devices) = self.host.output_devices() {
            for device in output_devices {
                if let Ok(name) = device.name() {
                    if name == device_name {
                        return Some(device);
                    }
                }
            }
        }

        None
    }

    // Start audio processing
    fn start_processing(&mut self) -> Result<(), String> {
        let mut state = self.audio_state.lock().unwrap();

        if state.running {
            return Ok(());
        }

        // Get selected devices
        let input_device_index = state.input_device_index;
        let output_device_index = state.output_device_index;

        drop(state); // Release lock before starting streams

        // Get input device
        let input_device = self
            .get_input_device(input_device_index)
            .ok_or_else(|| "Failed to get input device".to_string())?;

        // Get output device
        let output_device = self
            .get_output_device(output_device_index)
            .ok_or_else(|| "Failed to get output device".to_string())?;

        // Get default configs for devices
        let input_config = input_device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {}", e))?;

        let output_config = output_device
            .default_output_config()
            .map_err(|e| format!("Failed to get output config: {}", e))?;

        // Update sample rate in state
        {
            let mut state = self.audio_state.lock().unwrap();
            state.sample_rate = input_config.sample_rate().0;
        }

        // Configure input stream
        let input_stream_config = StreamConfig {
            channels: CHANNELS as u16,
            sample_rate: input_config.sample_rate(),
            buffer_size: BufferSize::Fixed(BUFFER_SIZE as u32),
        };

        // Configure output stream
        let output_stream_config = StreamConfig {
            channels: CHANNELS as u16,
            sample_rate: output_config.sample_rate(),
            buffer_size: BufferSize::Fixed(BUFFER_SIZE as u32),
        };

        // Create a buffer with capacity for a few frames of audio
        let buffer_size = BUFFER_SIZE * CHANNELS * 4;
        let buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(buffer_size)));
        self.audio_buffer = Some(buffer.clone());

        // Store weak reference to audio state for closures
        let audio_state_weak_input = Arc::downgrade(&self.audio_state);
        let audio_state_weak_output = Arc::downgrade(&self.audio_state);

        let spectrum_analyzer_weak = Arc::downgrade(&self.spectrum_analyzer);

        // Clone Arc for input closure
        let input_buffer = buffer.clone();

        // Build input stream
        let input_stream = match input_config.sample_format() {
            SampleFormat::F32 => {
                let input_stream = input_device
                    .build_input_stream(
                        &input_stream_config,
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            // Get lock on buffer
                            if let Ok(mut buffer) = input_buffer.lock() {
                                // Append input data to buffer, preserving channel interleaving
                                buffer.extend_from_slice(data);

                                // Trim buffer if it gets too large
                                if buffer.len() > buffer_size {
                                    let excess = buffer.len() - buffer_size;
                                    buffer.drain(0..excess);
                                }
                            }
                            
                            // Process data for spectrum analyzer
                            if let Some(analyzer) = spectrum_analyzer_weak.upgrade() {
                                if let Ok(mut analyzer) = analyzer.lock() {
                                    // Add samples to analyzer buffer
                                    if analyzer.add_samples(data) {
                                        // If we have enough data for FFT, process it
                                        if let Some(state) = audio_state_weak_input.upgrade() {
                                            if let Ok(mut state) = state.lock() {
                                                let sample_rate = state.sample_rate;
                                                let spectrum = analyzer.process_fft(sample_rate);
                                                
                                                // Apply some temporal smoothing
                                                for i in 0..SPECTRUM_POINTS {
                                                    state.spectrum_data[i] = state.spectrum_data[i] * SPECTRUM_DECAY
                                                        + spectrum[i] * (1.0 - SPECTRUM_DECAY);
                                                    
                                                    // Update peak values with slow decay
                                                    if state.spectrum_data[i] > state.spectrum_peak[i] {
                                                        state.spectrum_peak[i] = state.spectrum_data[i];
                                                    } else {
                                                        state.spectrum_peak[i] *= 0.98; // Slow decay
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        |err| eprintln!("Input stream error: {}", err),
                        None, // No timeouts
                    )
                    .map_err(|e| format!("Failed to build input stream: {}", e))?;

                input_stream
            }
            _ => {
                return Err("Unsupported sample format".to_string());
            }
        };

        // Start input stream
        input_stream
            .play()
            .map_err(|e| format!("Failed to start input stream: {}", e))?;

        // Store input stream
        self.input_stream = Some(input_stream);

        // Clone Arc for output closure
        let output_buffer = buffer;

        // Build output stream
        let output_stream = match output_config.sample_format() {
            SampleFormat::F32 => {
                let output_stream = output_device
                    .build_output_stream(
                        &output_stream_config,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            // Get lock on buffer
                            if let Ok(mut buffer) = output_buffer.lock() {
                                let mut i = 0;
                                
                                // Process samples ensuring we maintain stereo
                                while i < data.len() && buffer.len() >= CHANNELS {
                                    // For stereo, we need to process left and right separately
                                    for c in 0..CHANNELS.min(data.len() - i) {
                                        if !buffer.is_empty() {
                                            let sample = buffer.remove(0);
                                            
                                            // Apply EQ if state is available
                                            if let Some(state) = audio_state_weak_output.upgrade() {
                                                if let Ok(state) = state.lock() {
                                                    // Simple gain application using averaged gains for now
                                                    // In a real EQ you'd do frequency filtering here
                                                    let gain_sum: f32 = state.eq_gains.iter().sum();
                                                    let avg_gain = gain_sum / (state.eq_gains.len() as f32);
                                                    let gain_factor = 10.0_f32.powf(avg_gain / 20.0); // Convert dB to linear
                                                    data[i + c] = sample * gain_factor;
                                                } else {
                                                    data[i + c] = sample;
                                                }
                                            } else {
                                                data[i + c] = sample;
                                            }
                                        } else {
                                            data[i + c] = 0.0;
                                        }
                                    }
                                    i += CHANNELS;
                                }
                                
                                // Fill remaining buffer with silence if any
                                for j in i..data.len() {
                                    data[j] = 0.0;
                                }
                            } else {
                                // If we can't get lock, just play silence
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
                                }
                            }
                        },
                        |err| eprintln!("Output stream error: {}", err),
                        None, // No timeouts
                    )
                    .map_err(|e| format!("Failed to build output stream: {}", e))?;

                output_stream
            }
            _ => {
                return Err("Unsupported sample format".to_string());
            }
        };

        // Start output stream
        output_stream
            .play()
            .map_err(|e| format!("Failed to start output stream: {}", e))?;

        // Store output stream
        self.output_stream = Some(output_stream);

        // Update state
        let mut state = self.audio_state.lock().unwrap();
        state.running = true;

        Ok(())
    }

    // Stop audio processing
    fn stop_processing(&mut self) {
        // Stop streams
        self.input_stream = None;
        self.output_stream = None;

        // Clear audio buffer
        self.audio_buffer = None;

        // Update state
        let mut state = self.audio_state.lock().unwrap();
        state.running = false;
    }

    // Load preset EQ configuration
    fn load_preset(&mut self, preset_name: &str) {
        let mut state = self.audio_state.lock().unwrap();

        if let Some(preset) = state.presets.get(preset_name) {
            state.eq_gains = *preset;
        }
    }
    
    // Draw the spectrum analyzer
fn draw_spectrum(&self, ui: &mut egui::Ui) {
    ui.group(|ui| {
        ui.set_height(180.0);
        ui.heading("Spectrum Analyzer");
        
        let available_width = ui.available_width();
        let available_height = ui.available_height();

        let response = ui.allocate_rect(
            egui::Rect::from_min_size(
                ui.min_rect().min,
                egui::vec2(available_width, available_height),
            ),
            egui::Sense::hover(),
        );

        // Get a copy of the spectrum data to avoid lock during drawing
        let (spectrum_data, spectrum_peak, sample_rate) = {
            let state = self.audio_state.lock().unwrap();
            (
                state.spectrum_data.clone(),
                state.spectrum_peak.clone(),
                state.sample_rate,
            )
        };

        let painter = ui.painter();
        let rect = response.rect;
        
        // Background
        painter.rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgb(20, 20, 30),
        );
        
        // Draw grid lines
        let grid_color = egui::Color32::from_rgba_premultiplied(100, 100, 100, 100);
        
        // Horizontal grid lines (level indicators)
        for i in 0..=5 {
            let y = rect.top() + (i as f32 * rect.height() / 5.0);
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(1.0, grid_color),
            );
            
            // Label for dB scale
            let db = -i as f32 * 12.0;
            painter.text(
                egui::pos2(rect.left() + 5.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{:+.0} dB", db),
                egui::FontId::proportional(9.0),
                egui::Color32::from_rgb(180, 180, 180),
            );
        }
        
        // Vertical grid lines (frequency bands)
        let freq_labels = ["20Hz", "50Hz", "100Hz", "200Hz", "500Hz", "1kHz", "2kHz", "5kHz", "10kHz", "20kHz"];
        let frequencies: [f32; 10] = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
        
        for (i, (&freq, &label)) in frequencies.iter().zip(freq_labels.iter()).enumerate() {
            // Convert frequency to x position using logarithmic scale
            let log_min = 20.0_f32.log10();
            let log_max = 20000.0_f32.log10();
            let log_freq = freq.log10();
            let x_pos = rect.left() + rect.width() * (log_freq - log_min) / (log_max - log_min);
            
            if x_pos > rect.left() && x_pos < rect.right() {
                painter.line_segment(
                    [egui::pos2(x_pos, rect.top()), egui::pos2(x_pos, rect.bottom() - 12.0)],
                    egui::Stroke::new(1.0, grid_color),
                );
                
                // Label for frequency
                painter.text(
                    egui::pos2(x_pos, rect.bottom() - 5.0),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(9.0),
                    egui::Color32::from_rgb(180, 180, 180),
                );
            }
        }
        
        // Draw spectrum as filled polygon
        if !spectrum_data.is_empty() {
            // Create a gradient color for the bars
            let base_color = egui::Color32::from_rgb(30, 70, 140);
            let bright_color = egui::Color32::from_rgb(50, 120, 250);

            for i in 0..SPECTRUM_POINTS {
                // Calculate the actual frequency for this bin
                let freq = (i as f32) * (sample_rate as f32) / (2.0 * FFT_SIZE as f32);
                
                // Skip frequencies below 20Hz for better visibility
                if freq < 20.0 {
                    continue;
                }
                
                // Map frequency to logarithmic scale
                let log_min = 20.0_f32.log10();
                let log_max = 20000.0_f32.log10();
                let log_freq = freq.log10().max(log_min).min(log_max);
                let x_pos = rect.left() + rect.width() * (log_freq - log_min) / (log_max - log_min);
                
                // Skip points outside the visible area
                if x_pos < rect.left() || x_pos > rect.right() {
                    continue;
                }
                
                // Calculate bar width (thinner at higher frequencies)
                let next_freq = ((i + 1) as f32) * (sample_rate as f32) / (2.0 * FFT_SIZE as f32);
                let next_log_freq = next_freq.log10().max(log_min).min(log_max);
                let next_x_pos = rect.left() + rect.width() * (next_log_freq - log_min) / (log_max - log_min);
                let bar_width = (next_x_pos - x_pos).max(1.0);
                
                // Scale magnitude to fit in the view (y-axis)
                let normalized_value = spectrum_data[i] / 60.0; // 0.0 to 1.0
                let bar_height = normalized_value * (rect.height() - 12.0);
                
                // Draw the bar
                if bar_height > 0.0 {
                    let bar_color = lerp_rgba(
                        egui::Rgba::from(base_color),
                        egui::Rgba::from(bright_color),
                        normalized_value,
                    );
                    
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::pos2(x_pos, rect.bottom() - 12.0 - bar_height),
                            egui::pos2(x_pos + bar_width, rect.bottom() - 12.0)
                        ),
                        0.0, // No rounding
                        egui::Color32::from(bar_color)
                    );
                }
            }
            
            // Draw peak line
            let mut peak_points = Vec::with_capacity(SPECTRUM_POINTS);
            for i in 0..SPECTRUM_POINTS {
                let freq = (i as f32) * (sample_rate as f32) / (2.0 * FFT_SIZE as f32);
                if freq < 20.0 {
                    continue;
                }
                
                // Map frequency to logarithmic scale
                let log_min = 20.0_f32.log10();
                let log_max = 20000.0_f32.log10();
                let log_freq = freq.log10().max(log_min).min(log_max);
                let x_pos = rect.left() + rect.width() * (log_freq - log_min) / (log_max - log_min);
                
                if x_pos < rect.left() || x_pos > rect.right() {
                    continue;
                }
                
                // Scale magnitude to fit in the view (y-axis)
                let normalized_value = spectrum_peak[i] / 60.0; // 0.0 to 1.0
                let y = rect.bottom() - 12.0 - normalized_value * (rect.height() - 12.0);
                
                peak_points.push(egui::pos2(x_pos, y));
            }
            
            // Draw peak line as a series of small points
            for point in &peak_points {
                painter.circle_filled(
                    *point,
                    1.5,
                    egui::Color32::from_rgb(255, 100, 100)
                );
            }
        }
    });
}
}

// Implement the eframe App trait for our application
impl App for SystemWideEQ {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let running = {
            let state = self.audio_state.lock().unwrap();
            state.running
        };

        // Request continuous redraws while running to update the spectrum analyzer
        if running {
            ctx.request_repaint();
        }

        // Top panel for controls
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
                    .button(if running { "Stop EQ" } else { "Start EQ" })
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
            });
        });

        // Side panel for device selection
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

            // Update device indices if they changed
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

        // Main central area for EQ sliders
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Equalizer");

            self.draw_spectrum(ui);

            // Simple EQ visualization
            ui.group(|ui| {
                ui.set_height(120.0);
                let available_width = ui.available_width();
                let available_height = ui.available_height();

                let response = ui.allocate_rect(
                    egui::Rect::from_min_size(
                        ui.min_rect().min,
                        egui::vec2(available_width, available_height),
                    ),
                    egui::Sense::hover(),
                );

                // Get a copy of the EQ gains to avoid lock during drawing
                let eq_gains = {
                    let state = self.audio_state.lock().unwrap();
                    state.eq_gains
                };

                // Draw the response graph using primitives
                let painter = ui.painter();
                let rect = response.rect;
                let center_y = rect.center().y;
                let height_scale = rect.height() / 24.0; // -12 to +12 dB

                // Draw zero line
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), center_y),
                        egui::pos2(rect.right(), center_y),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 100, 100)),
                );

                // Draw EQ points and connect them
                let mut points = Vec::with_capacity(BANDS);
                for (i, gain) in eq_gains.iter().enumerate() {
                    let x = rect.left() + (i as f32 / (BANDS - 1) as f32) * rect.width();
                    let y = center_y - gain * height_scale;
                    points.push(egui::pos2(x, y));
                }

                // Draw segments between points
                for i in 0..points.len() - 1 {
                    painter.line_segment(
                        [points[i], points[i + 1]],
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 250)),
                    );
                }

                // Draw points as circles
                for point in points {
                    painter.circle(
                        point,
                        4.0,
                        egui::Color32::from_rgb(100, 150, 250),
                        egui::Stroke::new(1.0, egui::Color32::WHITE),
                    );
                }

                // Draw frequency labels
                for (i, &freq) in FREQUENCIES.iter().enumerate() {
                    let x = rect.left() + (i as f32 / (BANDS - 1) as f32) * rect.width();
                    painter.text(
                        egui::pos2(x, rect.bottom() - 10.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{}", freq),
                        egui::FontId::default(),
                        egui::Color32::WHITE,
                    );
                }
            });

            // EQ sliders
            ui.horizontal(|ui| {
                for (i, &freq) in FREQUENCIES.iter().enumerate() {
                    ui.vertical(|ui| {
                        let mut gain = {
                            let state = self.audio_state.lock().unwrap();
                            state.eq_gains[i]
                        };

                        let response = ui.add(
                            egui::Slider::new(&mut gain, -12.0..=12.0)
                                .orientation(egui::SliderOrientation::Vertical)
                                .text(""),
                        );

                        if response.changed() {
                            let mut state = self.audio_state.lock().unwrap();
                            state.eq_gains[i] = gain;
                        }

                        ui.label(format!("{} Hz", freq));
                    });
                }
            });
        });
    }
}

fn main() {
    // Initialize native options with the appropriate fields
    let options = NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "System-Wide Equalizer",
        options,
        Box::new(|cc| Ok(Box::new(SystemWideEQ::new(cc)))),
    )
    .expect("Failed to start application");
}
