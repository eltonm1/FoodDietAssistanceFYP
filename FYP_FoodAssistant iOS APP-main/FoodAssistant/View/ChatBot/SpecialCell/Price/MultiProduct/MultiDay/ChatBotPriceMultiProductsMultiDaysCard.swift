//
//  ChatBotPriceMultiProductsMultiDaysCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.MultipleProductsCell {
    struct MultiDaysCard: View {
        @Binding var path: NavigationPath
        let products: [Product]
        let sm: Supermarket
        var body: some View {
            let prices: [ProductPrice] = products.flatMap(\.prices).filter { $0.supermarket == sm }
            let sortedDates: [Date] = prices.compactMap({ $0.date }).sorted(by: \.date)
            let firstDay: Date = sortedDates.first!.date
            let lastDay: Date = sortedDates.last!.date
            let colorLegends: [String : Color] = Dictionary(uniqueKeysWithValues:
                                                                products.map { (product: Product) -> (String, Color) in
                (product.id.description, .random)
            })
            VStack(alignment: .leading, spacing: 8) {
                Text("\(firstDay.formatted(.dateTime.month().day())) - \(lastDay.formatted(.dateTime.month().day()))")
                    .productFont(.bold, relativeTo: .title)
                    .foregroundColor(.primary)
                ChatBotPriceStatistics(prices: prices)
                Chart(products) { (product: Product) in
                    ForEach(product.prices.filter( { $0.supermarket == sm } )) { (price: ProductPrice) in
                        LineMark(
                            x: .value("Date", price.date),
                            y: .value("price", price.price)
                        )
                        .interpolationMethod(.catmullRom)
                        .symbol(by: .value("product", product.name))
                        .foregroundStyle(by: .value("product", product.id.description))
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
                .chartYAxis() {
                    AxisMarks(position: .leading)
                }
                .chartXAxis() {
                    let duration: Int = lastDay.weekOfYear - firstDay.weekOfYear
                    let lessThanAWeek: Bool = duration < 1
                    let lessThanAMonth: Bool = duration <= 4
                    AxisMarks(
                        values: .stride(by: lessThanAWeek ? .day : (lessThanAMonth ? .weekOfYear : .month))
                    ) { (value: AxisValue) in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(
                            format: lessThanAWeek ? .dateTime.day() : (lessThanAMonth ? .dateTime.day().month() : .dateTime.month())
                        )
                    }
                }
                .frame(maxHeight: .infinity)
            }
            .previewContextMenu(
                destination: ChatBotProductList(products: products),
                preview: ChatBotProductList(products: products),
                navigationValue: products
            )
        }
    }
}

struct ChatBotPriceMultiProductsMultiDaysCard_Previews: PreviewProvider {
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
