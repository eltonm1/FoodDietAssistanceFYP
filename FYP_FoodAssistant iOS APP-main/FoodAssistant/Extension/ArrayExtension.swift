//
//  ArrayExtension.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 13/1/2023.
//

import Foundation

extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0, index < endIndex else {
            return nil
        }
        return self[index]
    }
}
