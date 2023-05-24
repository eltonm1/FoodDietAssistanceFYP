//
//  ChatBotMessageView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/3/2023.
//

import SwiftUI

struct ChatBotMessageView: View {
    @Binding var path: NavigationPath
    let ns: Namespace.ID
    let viewWidth: CGFloat
    var message: ChatBotMessage
    
    var body: some View {
        if message.isBot {
            VStack(alignment: .leading) {
                Text(message.response ?? "")
                    .lineLimit(nil)
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.white)
                    .padding(10)
                    .background(.blue)
                    .cornerRadius(10)
                    .frame(width: viewWidth * 3/4, alignment: .leading)
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(.leading)
                ChatBotSpecialMessageView(
                    path: $path,
                    ns: ns,
                    message: message,
                    viewWidth: viewWidth
                )
            }
            .frame(width: viewWidth, alignment: .leading)
        } else {
            ChatBotClientMessageView(message: message.clientInput)
                .frame(width: viewWidth, alignment: .trailing)
                .padding(.trailing)
        }
    }
}
