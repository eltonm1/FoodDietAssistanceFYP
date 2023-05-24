//
//  ChatBotProductPriceCell.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/3/2023.
//

import Charts
import SwiftUI

extension ChatBotSpecialMessageView {
    struct PriceCell: View {
        @Binding var path: NavigationPath
        let message: ChatBotMessage
        let isSingleProduct: Bool
        let viewWidth: CGFloat
        
        init(
            message: ChatBotMessage,
            viewWidth: CGFloat,
            path: Binding<NavigationPath>
        ) {
            self.message = message
            self.isSingleProduct = message.productsResponse.count == 1
            self.viewWidth = viewWidth
            self._path = path
        }
        
        var body: some View {
            if isSingleProduct {
                SingleProductCell(
                    product: message.productsResponse.first!,
                    viewWidth: viewWidth
                )
            } else {
                MultipleProductsCell(
                    products: message.productsResponse,
                    viewWidth: viewWidth,
                    path: self._path
                )
            }
        }
    }
}

struct ChatBotProductPriceCell_PreviewProvider: PreviewProvider {
    @State static var path = NavigationPath()
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let width: CGFloat = proxy.size.width
            let height: CGFloat = proxy.size.height
            ChatBotSpecialMessageView
                .PriceCell(message: dummyMessages[0], viewWidth: width, path: $path)
                .frame(height: height, alignment: .center)
        }
        .frame(maxHeight: .infinity, alignment: .center)
        .preferredColorScheme(.dark)
    }
}
