//
//  ChatBotProductDetailView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/3/2023.
//

import SwiftUI

extension ChatBotSpecialMessageView {
    struct DetailCell: View {
        let products: [Product]
        let viewWidth: CGFloat
        var body: some View {
            let count: CGFloat = CGFloat(products.count)
            let rowWidth: CGFloat = max(0, viewWidth - 48)
            let padding: CGFloat = 16
            let totalWidth: CGFloat = rowWidth*count + padding*max(0, count-1)
            let height: CGFloat = 100
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: padding) {
                    ForEach(products) { product in
                        ChatBotProductCell(
                            product: product,
                            color: .systemBlue,
                            width: rowWidth,
                            height: height
                        )
                    }
                }
                .frame(width: totalWidth)
                .padding(.horizontal)
            }
            .frame(width: viewWidth, height: height)
        }
    }
}

struct ChatBotProductDetailCell_Previews: PreviewProvider {
    static var previews: some View {
//        GeometryReader { (proxy: GeometryProxy) in
//            let width: CGFloat = proxy.size.width
//            let height: CGFloat = proxy.size.height
//            ChatBotProductDetailCell(products: dummyMessages[0].productsResponse, viewWidth: width)
//                .frame(height: height, alignment: .center)
//        }
//        .frame(maxHeight: .infinity, alignment: .center)
//        .preferredColorScheme(.dark)
        ChatBotView()
    }
}
