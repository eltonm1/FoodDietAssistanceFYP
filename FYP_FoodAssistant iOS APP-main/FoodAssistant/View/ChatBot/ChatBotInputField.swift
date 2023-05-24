//
//  ChatBotInputField.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/3/2023.
//

import SwiftUI
import SwiftUIX

struct ChatBotInputField: View {
    @Environment(\.safeAreaInsets) var inset
    @StateObject var keyboard = Keyboard()
    @Binding var message: String?
    @FocusState var isFocused: Bool
    let onSendButtonTapped: () -> Void
    let placeholder: String = "Message..."
    let gray: Color = .init(UIColor(red: 240/255, green: 240/255, blue: 240/255, alpha: 1.0))
    var isTextNilOrEmpty: Bool {
        message == nil || message?.isEmpty ?? true
    }
    var body: some View {
        HStack(spacing: 0) {
            TextField("", text: $message)
                .productFont(.regular, relativeTo: .body)
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    Text(placeholder)
                        .productFont(.regular, relativeTo: .body)
                        .foregroundColor(.secondary)
                        .opacity(isTextNilOrEmpty ? 1 : 0)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .focused($isFocused)
                .padding(8)
                .padding(.leading, 4)
            Button {
                isFocused = false
                onSendButtonTapped()
            } label: {
                Image(systemName: "arrow.up")
                    .productFont(.bold, relativeTo: .body)
                    .foregroundColor(.primary)
                    .padding(8)
                    .background(.systemOrange)
                    .clipShape(Circle())
                    .padding(.trailing, 8)
                    .padding(.vertical, 8)
            }
        }
        .overlay {
            Capsule()
                .strokeBorder(.primary)
        }
        .padding()
        .padding(.bottom, !keyboard.isShowing ? inset.bottom : 0)
        .background(.ultraThinMaterial, in: Rectangle())
        .offset(y: !keyboard.isShowing ? inset.bottom : 0)
    }
}

struct ChatBotInputField_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotView()
            .preferredColorScheme(.dark)
    }
}
