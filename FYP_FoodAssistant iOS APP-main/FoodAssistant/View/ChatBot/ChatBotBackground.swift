//
//  ChatBotBackground.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 19/3/2023.
//

import SwiftUI

struct ChatBotBackground: View {
    @Environment(\.colorScheme) var scheme
    let lightThemeColors: [Color] = [Color.blue.opacity(0.4), Color.systemBlue.opacity(0.8)]
    let darkThemeColors: [Color] = [Color.blue.opacity(0.6), Color.systemBlue.opacity(0.2)]
    var body: some View {
        LinearGradient(
            colors: scheme == .light ? lightThemeColors : darkThemeColors,
            startPoint: .top,
            endPoint: .bottom
        )
        .overlay {
            Circle()
                .foregroundColor(Color("GreenCircle"))
                .blur(radius: 100)
                .offset(x: -130, y: -100)
        }
        .overlay {
            Circle()
                .foregroundColor(Color("PinkCircle"))
                .blur(radius: 100)
                .offset(x: 130, y: 100)
        }
        .edgesIgnoringSafeArea(.all)
    }
}

struct ChatBotBackground_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotBackground()
            .preferredColorScheme(.light)
    }
}
