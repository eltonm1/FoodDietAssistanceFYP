//
//  CGSizePreferenceKey.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI

struct CGRectPreferenceKey: PreferenceKey {
    static var defaultValue: [String : CGRect] = [:]
    
    static func reduce(value: inout [String : CGRect], nextValue: () -> [String : CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}
