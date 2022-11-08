feature {
  name: "aid"
  type: INT
  int_domain {
    name: "aid"
    min: 0
    max: 1855610 
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
  value_count {
    min: 2
    max: 512
  }
}

feature {
  name: "target"
  type: INT
  int_domain {
    name: "target"
    min: 1
    max: 3 
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
  value_count {
    min: 2
    max: 512
  }
}
