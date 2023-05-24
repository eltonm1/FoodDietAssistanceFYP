//
//  SFButton.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import SwiftUI

struct SFButton: View {
    
    let sfSymbol: String
    let action: () -> Void
    
    init(
        _ sfSymbol: String,
        action: (() -> Void)? = nil
    ) {
        self.sfSymbol = sfSymbol
        self.action = action ?? { }
    }
    
    var body: some View {
        Button(action: action) {
            Image(systemName: sfSymbol)
                .resizable()
        }
    }
}

struct SFButton_Previews: PreviewProvider {
    static var previews: some View {
        SFButton("circle.fill")
    }
}
