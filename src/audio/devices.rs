use cpal::{Device, Host, StreamConfig, BufferSize};
use cpal::traits::{DeviceTrait, HostTrait};
use crate::config::BUFFER_SIZE;

pub fn get_input_device(host: &Host, devices: &[String], index: usize) -> Option<Device> {
    if index >= devices.len() {
        return None;
    }

    let device_name = &devices[index];
    if !device_name.starts_with("Input:") {
        return None;
    }

    let device_name = device_name.trim_start_matches("Input: ");

    if let Ok(input_devices) = host.input_devices() {
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

pub fn get_output_device(host: &Host, devices: &[String], index: usize) -> Option<Device> {
    if index >= devices.len() {
        return None;
    }

    let device_name = &devices[index];
    if !device_name.starts_with("Output:") {
        return None;
    }

    let device_name = device_name.trim_start_matches("Output: ");

    if let Ok(output_devices) = host.output_devices() {
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

pub fn init_devices(host: &Host) -> (Vec<String>, usize, usize) {
    let mut devices = Vec::new();
    let mut input_device_index = 0;
    let mut output_device_index = 0;

    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                devices.push(format!("Input: {}", name));
            }
        }
    }

    if let Ok(output_devices) = host.output_devices() {
        for device in output_devices {
            if let Ok(name) = device.name() {
                devices.push(format!("Output: {}", name));
            }
        }
    }

    // Set default devices
    if let Some(default_input) = host.default_input_device() {
        if let Ok(name) = default_input.name() {
            for (i, device_name) in devices.iter().enumerate() {
                if device_name.contains(&name) && device_name.starts_with("Input:") {
                    input_device_index = i;
                    break;
                }
            }
        }
    }

    if let Some(default_output) = host.default_output_device() {
        if let Ok(name) = default_output.name() {
            for (i, device_name) in devices.iter().enumerate() {
                if device_name.contains(&name) && device_name.starts_with("Output:") {
                    output_device_index = i;
                    break;
                }
            }
        }
    }

    (devices, input_device_index, output_device_index)
}

pub fn create_stream_config(channels: u16, sample_rate: cpal::SampleRate) -> StreamConfig {
    StreamConfig {
        channels,
        sample_rate,
        buffer_size: BufferSize::Fixed(BUFFER_SIZE as u32),
    }
}
