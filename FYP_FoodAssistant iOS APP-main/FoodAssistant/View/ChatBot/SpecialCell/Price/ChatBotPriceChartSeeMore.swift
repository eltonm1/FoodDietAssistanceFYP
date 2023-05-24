//
//  ChatBotPriceChartSeeMore.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 18/3/2023.
//

import SwiftUI
import Charts

extension View {
    func chatBotPriceChartSeeMore(
        products: [Product],
        colorLegends: [String : Color],
        action: (() -> Void)? = nil
    ) -> some View {
        self.modifier(
            ChatBotPriceChartSeeMoreModifier(
                products: products,
                colorLegends: colorLegends,
                action: action ?? { }
            )
        )
    }
}

struct ChatBotPriceChartSeeMoreModifier: ViewModifier {
    let products: [Product]
    let colorLegends: [String : Color]
    let action: () -> Void
    func body(content: Content) -> some View {
        content
            .chartForegroundStyleScale { (id: String) in
                colorLegends[id]!
            }
            .chartLegend(position: .bottom) {
                let count: Int = products.count
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(0..<min(3, count), id: \.self) { (index: Int) in
                        let product: Product = products[index]
                        let id: String = product.id.description
                        let name: String = product.name
                        HStack {
                            Circle()
                                .fill(colorLegends[id]!)
                                .frame(width: 10, height: 10)
                            Text(name)
                                .productFont(.regular, relativeTo: .body)
                                .lineLimit(1)
                                .foregroundColor(.primary)
                        }
                        .blur(radius: count <= 2 ? 0 : CGFloat(index < 1 ? 0 : index*2))
                    }
                }
                .overlay(alignment: .bottom) {
                    if products.count >= 3 {
                        Button(action: action) {
                            Text("See More")
                                .productFont(.bold, relativeTo: .body)
                                .foregroundStyle(.primary)
                                .padding(.horizontal)
                                .padding(.vertical, 4)
                                .background(.thinMaterial, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .frame(maxWidth: .infinity)
            }
    }
}
