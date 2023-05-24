//
//  ChatBotPriceMultiProductsSingleSameDayMultiProductsCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.MultipleProductsCell.SingleDayCard {
    struct MultiProductsCard: View {
        @Binding var path: NavigationPath
        let products: [Product]
        let sm: Supermarket
        let prices: [ProductPrice]
        let colorLegends: [String : Color]
        var body: some View {
            let day: Date = prices.sorted(by: \.date).first!.date
            VStack(alignment: .leading, spacing: 8) {
                Text("\(day.formatted(.dateTime.month().day().year()))")
                    .productFont(.bold, relativeTo: .title)
                    .foregroundColor(.primary)
                ChatBotPriceStatistics(prices: prices)
                Chart {
                    ForEach(products) { (product: Product) in
                        ForEach(product.prices.filter( { $0.supermarket == sm} )) { (price: ProductPrice) in
                            BarMark(
                                x: .value("name", product.name),
                                y: .value("price", price.price)
                            )
                            .foregroundStyle(by: .value("product", product.id.description))
                        }
                    }
                }
                .chatBotPriceChartSeeMore(
                    products: products,
                    colorLegends: colorLegends
                ) {
                    path.append(products)
                }
                .chartYAxisLabel(position: .top, spacing: 8) {
                    Text("Price (HKD)")
                        .productFont(.regular, relativeTo: .subheadline)
                        .foregroundColor(.secondary)
                }
                .chartXAxisLabel(position: .bottom, alignment: .center, spacing: 4) {
                    Text("Product")
                        .productFont(.regular, relativeTo: .subheadline)
                        .foregroundColor(.secondary)
                }
                .chartYAxis() {
                    AxisMarks(position: .leading)
                }
                .chartXAxis(.hidden)
            }
            .previewContextMenu(
                destination: ChatBotProductList(products: products),
                preview: ChatBotProductList(products: products),
                navigationValue: products
            )
        }
    }
}

struct ChatBotPriceMultiProductsSingleSameDayMultiProductsCard_Previews: PreviewProvider {
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
