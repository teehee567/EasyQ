use eframe::egui;
use crate::config::FREQUENCIES;

pub fn lerp_rgba(from: egui::Rgba, to: egui::Rgba, t: f32) -> egui::Rgba {
    egui::Rgba::from_rgba_premultiplied(
        from.r() + (to.r() - from.r()) * t,
        from.g() + (to.g() - from.g()) * t,
        from.b() + (to.b() - from.b()) * t,
        from.a() + (to.a() - from.a()) * t,
    )
}

pub fn draw_spectrum(
    ui: &mut egui::Ui, 
    spectrum_data: &[f32], 
    spectrum_peak: &[f32], 
) {
    ui.group(|ui| {
        ui.set_height(180.0);
        ui.heading("Spectrum Analyzer");
        
        // margins
        let left_margin = 45.0;
        let bottom_margin = 15.0;
        
        let available_width = ui.available_width();
        let available_height = ui.available_height();

        let response = ui.allocate_rect(
            egui::Rect::from_min_size(
                ui.min_rect().min,
                egui::vec2(available_width, available_height),
            ),
            egui::Sense::hover(),
        );

        let painter = ui.painter();
        let rect = response.rect;
        
        // Calculate Graph Area
        let graph_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left() + left_margin, rect.top()),
            egui::pos2(rect.right(), rect.bottom() - bottom_margin),
        );
        
        // Background
        painter.rect_filled(
            rect,
            5.0,
            egui::Color32::from_rgb(20, 20, 30),
        );
        
        // Draw grid lines
        let grid_color = egui::Color32::from_rgba_premultiplied(100, 100, 100, 100);
        
        // Level Indicators
        for i in 0..=5 {
            let y = graph_rect.top() + (i as f32 * graph_rect.height() / 5.0);
            painter.line_segment(
                [egui::pos2(graph_rect.left(), y), egui::pos2(graph_rect.right(), y)],
                egui::Stroke::new(1.0, grid_color),
            );
            
            // Label
            let db = -i as f32 * 12.0;
            painter.text(
                egui::pos2(rect.left() + 5.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{:+.0} dB", db),
                egui::FontId::proportional(9.0),
                egui::Color32::from_rgb(180, 180, 180),
            );
        }
        
        let freq_labels = ["20Hz", "50Hz", "100Hz", "200Hz", "500Hz", "1kHz", "2kHz", "5kHz", "10kHz", "20kHz"];
        
        for (&freq, &label) in FREQUENCIES.iter().zip(freq_labels.iter()) {
            let log_min = 20f32.log10();
            let log_max = 20000f32.log10();
            let log_freq = freq.log10();
            let x_pos = graph_rect.left() + graph_rect.width() * (log_freq - log_min) / (log_max - log_min);
            
            if x_pos > graph_rect.left() && x_pos < graph_rect.right() {
                painter.line_segment(
                    [egui::pos2(x_pos, graph_rect.top()), egui::pos2(x_pos, graph_rect.bottom())],
                    egui::Stroke::new(1.0, grid_color),
                );
                
                painter.text(
                    egui::pos2(x_pos, rect.bottom() - 5.0),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(9.0),
                    egui::Color32::from_rgb(180, 180, 180),
                );
            }
        }
        
        if !spectrum_data.is_empty() {
            let base_color = egui::Color32::from_rgb(30, 70, 140);
            let bright_color = egui::Color32::from_rgb(50, 120, 250);
            
            let bar_width = graph_rect.width() / (spectrum_data.len() as f32);
            
            for i in 0..spectrum_data.len() {
                let x_pos = graph_rect.left() + (i as f32 / (spectrum_data.len() - 1) as f32) * graph_rect.width();
                
                let normalized_value = spectrum_data[i] / 60.0; // 0.0 to 1.0
                let bar_height = normalized_value * graph_rect.height();
                
                if bar_height > 0.0 {
                    let bar_color = lerp_rgba(
                        egui::Rgba::from(base_color),
                        egui::Rgba::from(bright_color),
                        normalized_value,
                    );
                    
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::pos2(x_pos - bar_width/2.0, graph_rect.bottom() - bar_height),
                            egui::pos2(x_pos + bar_width/2.0, graph_rect.bottom())
                        ),
                        0.0,
                        egui::Color32::from(bar_color)
                    );
                }
                
                // Draw the peak dot
                if i < spectrum_peak.len() {
                    let peak_normalized_value = spectrum_peak[i] / 60.0; 
                    let peak_y = graph_rect.bottom() - peak_normalized_value * graph_rect.height();
                    
                    painter.circle_filled(
                        egui::pos2(x_pos, peak_y),
                        1.5,
                        egui::Color32::from_rgb(255, 100, 100)
                    );
                }
            }
        }
    });
}
