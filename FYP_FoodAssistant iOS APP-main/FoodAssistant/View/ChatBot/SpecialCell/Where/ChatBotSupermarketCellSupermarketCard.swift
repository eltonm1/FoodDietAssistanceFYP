//
//  ChatBotSupermarketCellSupermarketCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.SupermarketCell {
    struct SuperMarketCard: View {
        let products: [Product]
        let sm: Supermarket
        var body: some View {
            let count: Int = products.count
            let smName: String = sm.rawValue
            HStack {
                VStack(alignment: .leading) {
                    Text(smName)
                        .productFont(.bold, relativeTo: .title3)
                        .foregroundColor(.primary)
                    Text("\(count) product\(count > 1 ? "s" : "")")
                        .productFont(.regular, relativeTo: .body)
                        .foregroundColor(.secondary)
                }
                Spacer()
                HStack(spacing: -8) {
                    let num: Int = min(6, products.count)
                    ForEach(0..<num, id: \.self) { (index: Int) in
                        let product: Product = products[index]
                        ChatBotSpecialMessageView
                            .SupermarketCell
                            .ProductIcon(name: product.name)
                            .zIndex(Double(num - index))
                    }
                }
            }
        }
    }
}

struct ChatBotSupermarketCellSupermarketCard_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotView()
    }
}
