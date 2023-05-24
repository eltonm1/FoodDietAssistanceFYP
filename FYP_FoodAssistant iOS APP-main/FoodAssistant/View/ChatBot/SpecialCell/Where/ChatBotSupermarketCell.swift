//
//  ChatBotSupermarketCell.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 19/3/2023.
//

import SwiftUI

extension ChatBotSpecialMessageView {
    struct SupermarketCell: View {
        @Environment(\.colorScheme) var scheme
        let supermarkets: [Supermarket]
        let products: [Product]
        let viewWidth: CGFloat
        
        init(products: [Product], viewWidth: CGFloat) {
            self.products = products
            self.supermarkets = Array(Set(products.flatMap(\.prices).map(\.supermarket))).sorted(by: \.rawValue)
            self.viewWidth = viewWidth
        }
        
        var body: some View {
            let numRow: CGFloat = supermarkets.count > 1 ? 2 : 1
            let rowSpacing: CGFloat = 12
            let rowHeight: CGFloat = 100
            let totalHeight: CGFloat = numRow == 1 ? rowHeight : rowHeight*2 + rowSpacing
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHGrid(
                    rows: Array(repeating: GridItem(.fixed(rowHeight), spacing: rowSpacing), count: 2),
                    spacing: rowSpacing
                ) {
                    ForEach(supermarkets, id: \.rawValue) { (sm: Supermarket) in
                        let productsAtThatSM: [Product] = products.filter({ $0.prices.contains(where: { $0.supermarket == sm }) })
                        SuperMarketCard(products: productsAtThatSM, sm: sm)
                            .padding(20)
                            .frame(width: max(0, viewWidth - 48), height: rowHeight, alignment: .leading)
                            .background(Color.black.brightness(scheme == .light ? 0.95 : 0.2))
                            .cornerRadius(20, style: .continuous)
                            .previewContextMenu(
                                destination: ChatBotProductList(products: productsAtThatSM),
                                preview: ChatBotProductList(products: productsAtThatSM),
                                navigationValue: products
                            )
                    }
                }
                .padding(.horizontal)
            }
            .frame(height: totalHeight)
        }
    }
}

struct ChatBotSupermarketCell_Previews: PreviewProvider {
    static var previews: some View {
//        GeometryReader { (proxy: GeometryProxy) in
//            let width: CGFloat = proxy.size.width
//            let height: CGFloat = proxy.size.height
//            ChatBotSupermarketCell(products: dummyMessages[0].productsResponse, viewWidth: width)
//                .frame(height: height, alignment: .center)
//        }
//        .frame(maxHeight: .infinity, alignment: .center)
//        .preferredColorScheme(.dark)
        ChatBotView()
    }
}
