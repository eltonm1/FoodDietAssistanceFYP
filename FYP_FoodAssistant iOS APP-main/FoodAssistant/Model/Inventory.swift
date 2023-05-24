//
//  Inventory.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 25/3/2023.
//

import SwiftUI
import Foundation

struct Inventory: IdentifyEquateCodeHashable {
    let id = UUID()
    var name: String
    let color: Color
    var description: String?
    var products: [Product]
    
    init() {
        name = "Untitled List"
        description = nil
        color = .random
        products = []
    }
    
    init(
        name: String,
        description: String?,
        products: [Product],
        color: Color = .random
    ) {
        self.name = name
        self.description = description
        self.products = products
        self.color = color
    }
}
