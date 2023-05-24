//
//  BlurMaterialView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 27/1/2023.
//

import Foundation
import UIKit
import SwiftUI

struct BlurMaterialView: UIViewRepresentable {

    let material: UIBlurEffect.Style

    init(_ material: UIBlurEffect.Style) {
        self.material = material
    }

    func makeUIView(context: Context) -> UIVisualEffectView {
        UIVisualEffectView(effect: UIBlurEffect(style: material))
    }

    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {
        uiView.effect = UIBlurEffect(style: material)
    }
}
