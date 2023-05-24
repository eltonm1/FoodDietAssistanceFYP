//
//  ChatBotClientMessageView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/3/2023.
//

import SwiftUI

struct ChatBotClientMessageView: View {
    let message: String
    let gray: Color = .init(UIColor(red: 240/255, green: 240/255, blue: 240/255, alpha: 1.0))
    var body: some View {
        Text(message)
            .productFont(.regular, relativeTo: .body)
            .foregroundColor(.black)
            .padding(10)
            .background(gray)
            .cornerRadius(10)
    }
}

