//
//  InventoryView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 25/3/2023.
//

import SwiftUI
import Charts

struct InventoryView: View {
    @StateObject var vm: InventoryViewModel
    let screenSize: CGSize
    init(
        screenSize: CGSize,
        dataSource: FoodProductDataService
    ) {
        self.screenSize = screenSize
        self._vm = StateObject(wrappedValue: InventoryViewModel(dataSource: dataSource))
    }
    var body: some View {
        NavigationStack {
            VStack {
                if vm.inventories.isEmpty {
                    NoInventoryView()
                } else {
                    InventoryListView(
                        searchingText: $vm.searchingText,
                        editingInventory: $vm.editingInventory,
                        inventories: vm.inventories,
                        summaryType: vm.summaryType,
                        onRefresh: vm.updateInventories
                    )
                }
            }
            .navigationTitle("Inventory")
            .productLargeNavigationBar()
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    MenuBarItems(
                        editingInventory: $vm.editingInventory,
                        summaryType: $vm.summaryType
                    )
                }
            }
        }
        .frame(width: screenSize.width, height: screenSize.height)
        .sheet(
            isPresented: Binding<Bool>(
                get: { vm.editingInventory != nil },
                set: { _ in }
            )
        ) {
            if let editingInventory: Inventory = vm.editingInventory {
                if let i: Int = vm.inventories.firstIndex(of: editingInventory) {
                    vm.inventories[i] = editingInventory
                } else {
                    vm.inventories.append(editingInventory)
                }
            }
            vm.editingInventory = nil
            vm.updateInventories()
        } content: {
            EditInventoryView(
                inventory: $vm.editingInventory,
                inventories: $vm.inventories,
                dataSource: vm.foodDataService
            )
        }

    }
}

fileprivate struct InventoryListView: View {
    @Binding var searchingText: String
    @Binding var editingInventory: Inventory?
    let inventories: [Inventory]
    let summaryType: InventoryViewModel.SummaryCategory?
    let onRefresh: () -> Void
    var filteredInventories: [Inventory] {
        if searchingText.isEmpty {
            return inventories
        } else {
            return inventories.filter({
                $0.name.lowercased().contains(searchingText.lowercased())
            })
        }
    }
    var body: some View {
        List(filteredInventories) { (inventory: Inventory) in
            InventoryCell(
                editingInventory: $editingInventory,
                inventory: inventory,
                summaryType: summaryType
            )
        }
        .searchable(text: $searchingText, prompt: "e.g. BBQ List")
        .refreshable {
            onRefresh()
        }
        .animation(.easeInOut, value: filteredInventories)
    }
}

fileprivate struct InventoryCell: View {
    @Binding var editingInventory: Inventory?
    let inventory: Inventory
    let summaryType: InventoryViewModel.SummaryCategory?
    var body: some View {
        let products: [Product] = inventory.products
        let count: Int = products.count
        Button {
            editingInventory = inventory
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 16) {
                    Image(systemName: "list.bullet.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .foregroundColor(inventory.color)
                        .frame(width: 20, height: 20)
                    VStack(alignment: .leading) {
                        Text(inventory.name)
                            .productFont(.bold, relativeTo: .headline)
                            .foregroundColor(.primary)
                        if let description: String = inventory.description {
                            Text(description)
                                .productFont(.regular, relativeTo: .body)
                                .foregroundColor(.primary)
                                .padding(.bottom, 8)
                        }
                        ForEach(0..<min(3, count), id: \.self) { (i: Int) in
                            let name: String = products[i].name
                            Text("- \(name)")
                                .productFont(.regular, relativeTo: .body)
                                .lineLimit(1)
                                .foregroundColor(.secondary)
                        }
                        if count > 3 {
                            Text("...")
                                .productFont(.regular, relativeTo: .body)
                                .foregroundColor(.secondary)
                        }
                    }
                    .animation(.easeInOut, value: products)
                }
                if let summaryType {
                    Graph(
                        products: products,
                        summaryType: summaryType
                    )
                }
            }
        }
        .padding(.top)
        .animation(.easeInOut, value: summaryType)
        .listRowInsets(
            EdgeInsets(
                top: 0,
                leading: 16,
                bottom: 16,
                trailing: 16
            )
        )
    }
}

fileprivate struct Graph: View {
    @State private var isAppeared: Bool = false
    let products: [Product]
    let summaryType: InventoryViewModel.SummaryCategory
    var datas: [Double] {
        var data: [Double]
        let products: [Product] = products.sorted(by: \.id)
        switch summaryType {
        case .energy:
            data = products.compactMap(\.nutrition?.energy).compactMap({Double($0)})
        case .sugar:
            data = products.compactMap(\.nutrition?.sugars)
        case .carbohydrates:
            data = products.compactMap(\.nutrition?.carbohydrates)
        }
        return data.filter({ $0 > 0 })
    }
    var body: some View {
        let grams: [Double] = datas
        if !grams.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                let average: Double = grams.reduce(Double.zero, +)/Double(grams.count)
                Text("\(summaryType.rawValue) Consumption")
                    .productFont(.bold, relativeTo: .headline)
                    .foregroundColor(.primary)
                Chart(0..<datas.count, id: \.self) { (index: Int) in
                    BarMark(
                        x: .value("index", "\(index)"),
                        y: .value("data", datas[index])
                    )
                    .foregroundStyle(.gray.opacity(0.3))
                    RuleMark(y: .value("average", average))
                        .annotation(position: .top, alignment: .leading) {
                            Text("Average: \(average.formatted())g")
                                .productFont(.bold, relativeTo: .headline)
                                .foregroundColor(.systemBlue)
                        }
                        .lineStyle(StrokeStyle(lineWidth: 3))
                }
                .chartLegend(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading) {
                        AxisGridLine().foregroundStyle(.clear)
                    }
                }
                .chartXAxis {
                    AxisMarks(position: .bottom) { _ in
                        AxisGridLine().foregroundStyle(.clear)
                    }
                }
                Text("\(grams.count)-product span")
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
            }
            .frame(maxHeight: isAppeared ? .infinity : 0)
            .animation(.easeInOut, value: isAppeared)
            .onAppear { isAppeared = true }
            .onDisappear { isAppeared = false }
        }
    }
}

fileprivate struct MenuBarItems: View {
    @Binding var editingInventory: Inventory?
    @Binding var summaryType: InventoryViewModel.SummaryCategory?
    var body: some View {
        Menu {
            Button {
                editingInventory = Inventory()
            } label: {
                Label("Add Inventory", systemImage: "plus")
            }
            Menu {
                ForEach(InventoryViewModel.SummaryCategory.allCases, id: \.self) {
                    (type: InventoryViewModel.SummaryCategory) in
                    let isSelected: Bool = summaryType == type
                    Button {
                        summaryType = type
                    } label: {
                        if isSelected {
                            Label(type.rawValue, systemImage: "checkmark")
                        } else {
                            Text(type.rawValue)
                        }
                    }
                }
                Divider()
                Button {
                    summaryType = nil
                } label: {
                    Label("Hide", systemImage: "eye.slash")
                }
                .foregroundColor(.systemRed)
            } label: {
                Label("Summary", systemImage: "info.circle")
            }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }
}

fileprivate struct NoInventoryView: View {
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let width: CGFloat = proxy.size.width
            let height: CGFloat = proxy.size.height
            ScrollView(showsIndicators: false) {
                VStack {
                    Image("empty")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 200)
                    Text("You don't have any inventory now")
                        .productFont(.regular, relativeTo: .body)
                        .foregroundColor(.primary)
                    Rectangle()
                        .foregroundColor(.clear)
                        .frame(height: height/8)
                }
                .frame(width: width, height: height)
            }
        }
    }
}

struct InventoryView_Previews: PreviewProvider {
    static var previews: some View {
        GeometryReader {
            let size: CGSize = $0.size
            NavigationStack {
                InventoryView(
                    screenSize: size,
                    dataSource: FoodProductDataService()
                )
            }
        }
    }
}
