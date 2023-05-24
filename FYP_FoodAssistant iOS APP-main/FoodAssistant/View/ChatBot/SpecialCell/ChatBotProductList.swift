//
//  ChatBotProductList.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI

struct ChatBotProductList: View {
    let products: [Product]
    let colors: [Product : Color]
    init(products: [Product]) {
        self.products = products
        self.colors = Dictionary(uniqueKeysWithValues: products.map {
            ($0, Color.random)
        })
    }
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let width: CGFloat = proxy.size.width
            ScrollView {
                VStack(spacing: 16) {
                    ForEach(products) { (product: Product) in
                        ChatBotProductCell(
                            product: product,
                            color: colors[product]!,
                            width: width - 32,
                            height: 100
                        )
                    }
                    .frame(height: 100)
                }
                .frame(width: width, alignment: .center)
            }
        }
    }
}

struct ChatBotProductList_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            ChatBotProductList(products: dummyProducts)
        }
    }
}
