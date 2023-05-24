//
//  ChatBotProductPriceSingleDayCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.MultipleProductsCell {
    struct SingleDayCard: View {
        @Binding var path: NavigationPath
        let products: [Product]
        let sm: Supermarket
        var body: some View {
            let prices: [ProductPrice] = products.flatMap(\.prices).filter { $0.supermarket == sm }
            let sortedDates: [Date] = prices.compactMap({ $0.date }).sorted(by: \.date)
            let firstDay: Date = sortedDates.first!.date
            let lastDay: Date = sortedDates.last!.date
            let colorLegends: [String : Color] = Dictionary(
                uniqueKeysWithValues: products.map { (product: Product) -> (String, Color) in
                (product.id.description, .random)
            })
            if firstDay.compare(toDate: lastDay, granularity: .day).rawValue == 0 {
                /// - Note: Same Day
                if prices.count == 1 {
                    SingleProductCard(product: products.first!)
                } else {
                    MultiProductsCard(
                        path: $path,
                        products: products,
                        sm: sm,
                        prices: prices,
                        colorLegends: colorLegends
                    )
                }
            } else {
                SingleDifferentDayCard(
                    path: $path,
                    products: products,
                    sm: sm,
                    prices: prices,
                    colorLegends: colorLegends,
                    firstDay: firstDay,
                    lastDay: lastDay
                )
            }
        }
    }
}

struct ChatBotProductPriceSingleDayCard_Previews: PreviewProvider {
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
