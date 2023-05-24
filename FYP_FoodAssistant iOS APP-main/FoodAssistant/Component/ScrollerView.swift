//
//  ScrollerView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 13/1/2023.
//

import SwiftUI

struct ScrollerView: View {
    let title: String
    let subtitle: String
    let color: Color
    let shouldDisappear: Bool
    let height: CGFloat
    let yOffset: CGFloat
    
    var body: some View {
        HStack {
            HStack {
                Text(title)
                    .productFont(.bold, relativeTo: .body)
                    .foregroundColor(.primary)
                Text(subtitle)
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
            }
            .padding()
            .padding(.trailing, 4)
            .background(.ultraThinMaterial)
            .clipShape(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .offset(x: shouldDisappear ? 200 : 0)
            .environment(\.colorScheme, .dark)
            
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(color)
                .frame(width: 2, height: height)
        }
        .padding(.trailing, 5)
        .offset(y: yOffset)
        .animation(.easeInOut, value: shouldDisappear)
    }
}
