//
//  ChatBotSingleProductManyDaysCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.SingleProductCell {
    struct ManyDaysCard: View {
        let prices: [ProductPrice]
        let curGradient = LinearGradient(
            gradient: Gradient (
                colors: [
                    .systemBlue.opacity(0.5),
                    .systemBlue.opacity(0.2),
                    .systemBlue.opacity(0.05),
                ]
            ),
            startPoint: .top,
            endPoint: .bottom
        )
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
                    LineMark(
                        x: .value("Date", price.date),
                        y: .value("price", price.price)
                    )
                    .interpolationMethod(.catmullRom)
                    .symbol() {
                        Circle()
                            .fill(.orange)
                            .frame(width: 10)
                    }
                    
                    AreaMark(
                        x: .value("Date", price.date),
                        y: .value("price", price.price)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(curGradient)
                    .symbol() {
                        Circle()
                            .fill(.orange)
                            .frame(width: 10)
                    }
                }
                .chartXAxisLabel(position: .bottom, alignment: .center) {
                    Text("Date (Day)")
                        .productFont(.regular, relativeTo: .subheadline)
                        .foregroundColor(.secondary)
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
                    AxisMarks(values: .stride(by: .weekOfYear)) {
                        (value: AxisValue) in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(
                            format: .dateTime.day().month()
                        )
                    }
                }
                .frame(maxHeight: .infinity)
            }
        }
    }
}

struct ChatBotSingleProductCellManyDaysCard_Previews: PreviewProvider {
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
