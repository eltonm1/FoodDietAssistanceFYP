//
//  ChatBotSingleProductCell.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 18/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell {
    struct SingleProductCell: View {
        @Environment(\.colorScheme) var scheme
        let product: Product
        let supermarkets: [Supermarket]
        let viewWidth: CGFloat
        let detail: InputProductDetailView.Detail
        
        init(product: Product, viewWidth: CGFloat) {
            self.product = product
            self.supermarkets = Array(Set(product.prices.map(\.supermarket))).sorted(by: \.rawValue)
            self.viewWidth = viewWidth
            self.detail = .init(product: product, editable: false)
        }
        
        var body: some View {
            let pricesGroupBySuperMarket: [[ProductPrice]] = supermarkets.map { (sm: Supermarket) -> [ProductPrice] in
                product.prices.filter { $0.supermarket == sm }
            }
            let maxHeight: Int = pricesGroupBySuperMarket
                .reduce(0) { (partialResult: Int, next: [ProductPrice]) -> Int in
                    max(partialResult, next.count > 1 ? 400 : 200)
                }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(supermarkets, id: \.self) { (sm: Supermarket) in
                        VStack(alignment: .leading) {
                            Text("\(sm.rawValue) Store")
                                .productFont(.bold, relativeTo: .body)
                                .foregroundColor(.secondary)
                            let pricesAtThatStore: [ProductPrice] = product.prices.filter { $0.supermarket == sm }
                            if pricesAtThatStore.count == 1 {
                                SingleDayCard(
                                    price: pricesAtThatStore[0],
                                    name: product.name
                                )
                            } else if pricesAtThatStore.count <= 7 {
                                WeekDaysCard(prices: pricesAtThatStore)
                            } else {
                                ManyDaysCard(prices: pricesAtThatStore)
                            }
                        }
                        .padding(32)
                        .frame(width: max(0, viewWidth - 48), height: CGFloat(maxHeight))
                        .background(Color.black.brightness(scheme == .light ? 0.95 : 0.2))
                        .cornerRadius(20, style: .continuous)
                        .previewContextMenu(
                            destination: InputProductDetailView(detail: detail),
                            preview: InputProductDetailView(detail: detail),
                            navigationValue: detail,
                            presentAsSheet: false
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct ChatBotSingleProductCell_Previews: PreviewProvider {
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
