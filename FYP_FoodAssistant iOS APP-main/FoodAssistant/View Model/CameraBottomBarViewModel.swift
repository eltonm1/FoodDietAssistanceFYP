//
//  CameraBottomBarViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/1/2023.
//

import Foundation
import SwiftUI

class CameraBottomBarViewModel: ObservableObject {
    @Published var showFlashLightLabel: Bool = false
    var scrollerTimeOut: CGFloat = 0
    
    func onTimerUpdate() {
        if scrollerTimeOut < 3 {
            scrollerTimeOut += 0.01
        } else {
            withAnimation(.spring()) {
                showFlashLightLabel = false
            }
        }
    }
    
    func onFlashLightModeToggle() {
        withAnimation(.spring()) {
            scrollerTimeOut = 0
            showFlashLightLabel = true
        }
    }
}
