//
//  ChatBotSingleProductWeekDaysCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.SingleProductCell {
    struct WeekDaysCard: View {
        let prices: [ProductPrice]
        var body: some View {
            let firstDay: Date = prices.sorted(by: \.date).first!.date
            let lastDay: Date = prices.sorted(by: \.date).last!.date
            VStack(alignment: .leading, spacing: 16) {
                Text("\(firstDay.formatted(.dateTime.month().day())) - \(lastDay.formatted(.dateTime.month().day()))")
                    .productFont(.bold, relativeTo: .title)
                    .foregroundColor(.primary)
                ChatBotPriceStatistics(prices: prices)
                    .padding(.bottom)
                Chart(prices) { (price: ProductPrice) in
                    BarMark(
                        x: .value("price", price.price),
                        y: .value("Date", price.date.formatted(.dateTime.day().month()))
                    )
                }
                .chartXAxisLabel(position: .bottom, alignment: .center) {
                    Text("Price (HKD)")
                        .productFont(.regular, relativeTo: .subheadline)
                        .foregroundColor(.secondary)
                }
                //            .chartYAxisLabel(position: .top, spacing: 8) {
                //                Text("Date")
                //                    .productFont(.bold, relativeTo: .subheadline)
                //                    .foregroundColor(.secondary)
                //            }
                .chartYAxis {
                    AxisMarks(preset: .extended) { (value: AxisValue) in
                        AxisGridLine()
                        AxisValueLabel {
                            let index: Int = value.index
                            let price: ProductPrice = prices[index]
                            Text("\(price.date.formatted(.dateTime.day().month()))")
                                .productFont(.regular, relativeTo: .footnote)
                        }
                    }
                }
            }
        }
    }
}

struct ChatBotSingleProductCellWeekDaysCard_Previews: PreviewProvider {
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
