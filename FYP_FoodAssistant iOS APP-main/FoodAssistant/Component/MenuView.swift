//
//  MenuView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/1/2023.
//

import SwiftUI

struct MenuView: View {
    let menuItem: MenuItem

    init(menuItem: MenuItem) {
        self.menuItem = menuItem
    }

    init(@MenuBuilder _ content: () -> MenuItem) {
        self.menuItem = content()
    }

    var body: some View {
        Menu {
            ForEach(menuItem.children, id: \.id) { (item: LabelItem) in
                if let menuItem = item as? MenuItem {
                    MenuView(menuItem: menuItem)
                } else if let buttonItem = item as? ButtonItem {
                    Button(action: buttonItem.action) {
                        if let label = buttonItem.label {
                            Label(label, systemImage: buttonItem.systemName)
                        } else {
                            Image(systemName: buttonItem.systemName)
                        }
                    }
                }
            }
        } label: {
            if let label = menuItem.label {
                Label(label, systemImage: menuItem.systemName)
            } else {
                Image(systemName: menuItem.systemName)
            }
        }
    }
}
