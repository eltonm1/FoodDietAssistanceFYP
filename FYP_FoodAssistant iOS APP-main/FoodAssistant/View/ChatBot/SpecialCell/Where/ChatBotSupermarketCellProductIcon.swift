//
//  ChatBotSupermarketCellProductIcon.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI

extension ChatBotSpecialMessageView.SupermarketCell {
    struct ProductIcon: View {
        let name: String
        var body: some View {
            Text(String(name.prefix(1)))
                .foregroundColor(.primary)
                .productFont(.bold, relativeTo: .body)
                .padding(8)
                .background(.random)
                .clipShape(Circle())
                .shadow(radius: 10)
        }
    }
}

struct ChatBotSupermarketCellProductIcon_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotView()
    }
}
