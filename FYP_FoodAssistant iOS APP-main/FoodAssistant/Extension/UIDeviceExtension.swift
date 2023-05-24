//
//  UIDeviceExtension.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 10/1/2023.
//

import UIKit

extension UIDevice {
    static var isIPad: Bool {
        UIDevice.current.userInterfaceIdiom == .pad
    }
    
    static var isIPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
}
