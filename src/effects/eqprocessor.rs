use std::vec::Vec;
use std::f32;
use std::f64;

use crate::config::BANDS;
use crate::config::FREQUENCIES;

const SMOOTHING_TIME_MS: f32 = 20.0;

#[derive(Clone, Debug)]
struct BiquadCoefficients {
    b0: f32, b1: f32, b2: f32, // Feedforward coef
    a1: f32, a2: f32, // Feedback coef
}

impl BiquadCoefficients {
    fn identity() -> Self {
        BiquadCoefficients { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 }
    }
}

struct BiquadFilter {
    coeffs: BiquadCoefficients, // smooth coef for processing
    target_coeffs: BiquadCoefficients,
    x1: f32, x2: f32, // Previous inputs
    y1: f32, y2: f32, // Previous outputs
    smoothing_factor: f32,
}

impl BiquadFilter {
    fn new(target_coeffs: BiquadCoefficients, sample_rate: f32) -> Self {
        let smoothing_factor = calculate_smoothing_factor(sample_rate, SMOOTHING_TIME_MS);
        BiquadFilter {
            coeffs: target_coeffs.clone(),
            target_coeffs,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
            smoothing_factor,
        }
    }

    fn set_target_coef(&mut self, target_coeffs: BiquadCoefficients) {
        self.target_coeffs = target_coeffs;
    }

    fn update_smoothing_factor(&mut self, sample_rate: f32) {
        self.smoothing_factor = calculate_smoothing_factor(sample_rate, SMOOTHING_TIME_MS);
    }

    #[inline]
    fn process_sample(&mut self, input: f32) -> f32 {
        // linear interpolation
        self.coeffs.b0 += (self.target_coeffs.b0 - self.coeffs.b0) * self.smoothing_factor;
        self.coeffs.b1 += (self.target_coeffs.b1 - self.coeffs.b1) * self.smoothing_factor;
        self.coeffs.b2 += (self.target_coeffs.b2 - self.coeffs.b2) * self.smoothing_factor;
        self.coeffs.a1 += (self.target_coeffs.a1 - self.coeffs.a1) * self.smoothing_factor;
        self.coeffs.a2 += (self.target_coeffs.a2 - self.coeffs.a2) * self.smoothing_factor;

        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        let output = self.coeffs.b0 * input + self.coeffs.b1 * self.x1 + self.coeffs.b2 * self.x2
                   - self.coeffs.a1 * self.y1 - self.coeffs.a2 * self.y2;

        self.x2 = self.x1; // x[n-2] = x[n-1]
        self.x1 = input; // x[n-1] = x[n]
        self.y2 = self.y1; // y[n-2] = y[n-1]
        self.y1 = output; // y[n-1] = y[n]

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
        self.coeffs = self.target_coeffs.clone();
    }
}


fn calculate_smoothing_factor(sample_rate: f32, smoothing_time_ms: f32) -> f32 {
    if sample_rate <= 0.0 || smoothing_time_ms <= 0.0 {
        return 1.0;
    }
    // T = smoothing_time_ms / 1000.0
    // alpha = 1 - exp(-2 * PI / (T * sample_rate))
    const TWO_PI: f64 = 2.0 * f64::consts::PI;
    let time_constant_sec = smoothing_time_ms as f64 * 0.001;
    (1.0 - (-TWO_PI / (time_constant_sec * sample_rate as f64)).exp()) as f32
}

// Q factor how much moving the band thing moves other bands around
fn calculate_q(center_freq: f32, next_freq: Option<f32>) -> f32 {
    const DEFAULT_Q: f32 = 1.414; // good default????

    match next_freq {
        Some(n_freq) => {
            if center_freq <= 0.0 || n_freq <= center_freq {
                return DEFAULT_Q;
            }
            let ratio = n_freq / center_freq;
            // Q = sqrt(ratio) / (ratio - 1)
            let q = ratio.sqrt() / (ratio - 1.0);
            q.max(0.1).min(20.0)
        },
        None => DEFAULT_Q,
    }
}


pub struct EQProcessor {
    filters: Vec<BiquadFilter>,
    current_gains: [f32; BANDS],
    sample_rate: f32,
}

impl EQProcessor {
    pub fn new(sample_rate: u32) -> Self {
        let sample_rate_f32 = sample_rate as f32;
        let mut filters = Vec::with_capacity(BANDS);
        let initial_gains = [0.0; BANDS];

        for i in 0..BANDS {
             let target_coeffs = calculate_intermediate_coef(
                i,
                sample_rate_f32,
                initial_gains[i],
             );
             filters.push(BiquadFilter::new(target_coeffs, sample_rate_f32));
        }

        Self {
            filters,
            current_gains: initial_gains,
            sample_rate: sample_rate_f32,
        }
    }

    pub fn update_gains(&mut self, new_gains: [f32; BANDS]) {
        for i in 0..BANDS {
            if (self.current_gains[i] - new_gains[i]).abs() > 1e-6 {
                self.current_gains[i] = new_gains[i];
                let target_coeffs = calculate_intermediate_coef(
                    i,
                    self.sample_rate,
                    self.current_gains[i],
                );
                self.filters[i].set_target_coef(target_coeffs);
            }
        }
    }

    pub fn update_gain(&mut self, band_index: usize, gain_db: f32) {
        if band_index < BANDS {
            if (self.current_gains[band_index] - gain_db).abs() > 1e-6 {
                self.current_gains[band_index] = gain_db;
                let target_coeffs = calculate_intermediate_coef(
                    band_index,
                    self.sample_rate,
                    gain_db,
                );
                self.filters[band_index].set_target_coef(target_coeffs);
            }
        } else {
            // lol
            panic!()
        }
    }

    pub fn update_sample_rate(&mut self, sample_rate: u32) {
        let sample_rate_f32 = sample_rate as f32;
        if (self.sample_rate - sample_rate_f32).abs() > 0.1 {
            self.sample_rate = sample_rate_f32;
            for i in 0..BANDS {
                 let target_coeffs = calculate_intermediate_coef(
                    i,
                    self.sample_rate,
                    self.current_gains[i],
                );
                 self.filters[i].update_smoothing_factor(self.sample_rate);
                 self.filters[i].set_target_coef(target_coeffs);
                 self.filters[i].reset();
            }
        }
    }

    pub fn process_block(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let mut current_sample = *sample;
            for filter in self.filters.iter_mut() {
                 current_sample = filter.process_sample(current_sample);
            }
            *sample = current_sample;
        }
    }

     pub fn process_sample(&mut self, sample: f32) -> f32 {
         let mut current_sample = sample;
         for filter in self.filters.iter_mut() {
             current_sample = filter.process_sample(current_sample);
         }
         current_sample
     }

     pub fn reset(&mut self) {
         for filter in self.filters.iter_mut() {
             filter.reset();
         }
     }
}
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//                      HEHEHEHAR
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
//https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
fn calculate_intermediate_coef(
    band_index: usize,
    sample_rate: f32,
    gain_db: f32,
) -> BiquadCoefficients {
    if band_index >= BANDS {
         return BiquadCoefficients::identity();
    }

    let freq = FREQUENCIES[band_index];

    // bypass if low gain
    if gain_db.abs() < 0.01 {
        return BiquadCoefficients::identity();
    }

    // use high shelf for last band
    if band_index == BANDS - 1 {
         let shelf_q = 1.0 / (2.0f32).sqrt();
         let (b0, b1, b2, a1, a2) = high_shelf_coef(
            sample_rate as f64,
            freq as f64,
            gain_db as f64,
            shelf_q as f64
        );
        BiquadCoefficients { b0, b1, b2, a1, a2 }
    } else {
        // use a peaking filter
        let next_freq_opt = if band_index < BANDS - 1 { Some(FREQUENCIES[band_index + 1]) } else { None };
        let q = calculate_q(freq, next_freq_opt);

        let (b0, b1, b2, a1, a2) = peaking_eq_coef(
            sample_rate as f64,
            freq as f64,
            q as f64,
            gain_db as f64
        );
        BiquadCoefficients { b0, b1, b2, a1, a2 }
    }
}


fn high_shelf_coef(sample_rate: f64, frequency: f64, gain_db: f64, q: f64) -> (f32, f32, f32, f32, f32) {
    let frequency = frequency.max(1.0).min(sample_rate * 0.499); // Stay below nyquist?????
    let q = q.max(0.01);

    let amp = 10.0f64.powf(gain_db / 20.0);
    let omega = 2.0 * f64::consts::PI * frequency / sample_rate;
    let cos_omega = omega.cos();
    let sin_omega = omega.sin();

    // alpha = sin(omega)/2 * sqrt( (A + 1/A)*(1/S - 1) + 2 )
    let alpha = sin_omega / (2.0 * q);

    let two_sqrt_a_alpha = 2.0 * amp.sqrt() * alpha;

    // high shelf coef
    let b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_omega + two_sqrt_a_alpha);
    let b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_omega);
    let b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_omega - two_sqrt_a_alpha);
    let a0 = (amp + 1.0) - (amp - 1.0) * cos_omega + two_sqrt_a_alpha;
    let a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_omega);
    let a2 = (amp + 1.0) - (amp - 1.0) * cos_omega - two_sqrt_a_alpha;

    // Normalize by a0
    if a0.abs() < 1e-10 {
        (1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32)
    } else {
        let inv_a0 = 1.0 / a0;
        ( (b0 * inv_a0) as f32, (b1 * inv_a0) as f32, (b2 * inv_a0) as f32,
          (a1 * inv_a0) as f32, (a2 * inv_a0) as f32 )
    }
}

fn peaking_eq_coef(sample_rate: f64, frequency: f64, q: f64, gain_db: f64) -> (f32, f32, f32, f32, f32) {
    let frequency = frequency.max(1.0).min(sample_rate * 0.499); // Stay below nyquist
    let q = q.max(0.01);

    let amp = 10.0f64.powf(gain_db / 40.0);
    let omega = 2.0 * f64::consts::PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();

    let alpha = (sin_omega / (2.0 * q)).max(1e-9);

    // peaking coef
    let b0 = 1.0 + alpha * amp;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * amp;
    let a0 = 1.0 + alpha / amp;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / amp;

    // Normalize by a0
    if a0.abs() < 1e-10 {
        (1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32)
    } else {
        let inv_a0 = 1.0 / a0;
        ( (b0 * inv_a0) as f32, (b1 * inv_a0) as f32, (b2 * inv_a0) as f32,
          (a1 * inv_a0) as f32, (a2 * inv_a0) as f32 )
    }
}

