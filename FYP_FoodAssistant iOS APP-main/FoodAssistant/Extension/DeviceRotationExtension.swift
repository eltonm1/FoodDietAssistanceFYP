//
//  DeviceRotationExtension.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 7/1/2023.
//

import Foundation
import SwiftUI

struct DeviceRotationViewModifier: ViewModifier {
    let action: (UIDeviceOrientation) -> Void

    func body(content: Content) -> some View {
        content
            .onAppear()
            .onReceive(
                NotificationCenter
                    .default
                    .publisher(for: UIDevice.orientationDidChangeNotification)
            ) { _ in
                action(UIDevice.current.orientation)
            }
    }
}

extension View {
    func onRotate(perform action: @escaping (UIDeviceOrientation) -> Void) -> some View {
        self.modifier(DeviceRotationViewModifier(action: action))
    }
}

extension UIScreen {
    static let protraitSize: CGSize = .init(
        width: min(UIScreen.main.bounds.size.width, UIScreen.main.bounds.size.height),
        height: max(UIScreen.main.bounds.size.width, UIScreen.main.bounds.size.height)
    )
    static let landscapeSize: CGSize = .init(
        width: max(UIScreen.main.bounds.size.width, UIScreen.main.bounds.size.height),
        height: min(UIScreen.main.bounds.size.width, UIScreen.main.bounds.size.height)
    )
}

