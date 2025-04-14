use eframe::egui;
use crate::config::{BANDS, FREQUENCIES};

pub fn draw_eq_visualization(ui: &mut egui::Ui, eq_gains: &[f32; BANDS]) {
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

        let painter = ui.painter();
        let rect = response.rect;
        let center_y = rect.center().y;
        let height_scale = rect.height() / 48.0;

        painter.line_segment(
            [
                egui::pos2(rect.left(), center_y),
                egui::pos2(rect.right(), center_y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 100, 100)),
        );

        // Draw EQ points
        let mut points = Vec::with_capacity(BANDS);
        for (i, gain) in eq_gains.iter().enumerate() {
            let x = rect.left() + (i as f32 / (BANDS - 1) as f32) * rect.width();
            let y = center_y - gain * height_scale;
            points.push(egui::pos2(x, y));
        }

        for i in 0..points.len() - 1 {
            painter.line_segment(
                [points[i], points[i + 1]],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 250)),
            );
        }

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
}

pub fn draw_eq_sliders(ui: &mut egui::Ui, eq_gains: &mut [f32; BANDS]) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        for (i, &freq) in FREQUENCIES.iter().enumerate() {
            ui.vertical(|ui| {
                let mut gain = eq_gains[i];

                let response = ui.add(
                    egui::Slider::new(&mut gain, -24.0..=24.0)
                        .orientation(egui::SliderOrientation::Vertical)
                        .text(""),
                );

                if response.changed() {
                    eq_gains[i] = gain;
                    changed = true;
                }

                ui.label(format!("{} Hz", freq));
            });
        }
    });

    changed
}
