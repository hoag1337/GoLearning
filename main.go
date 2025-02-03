package main

import (
	"fmt"
)

func longestMonotonicSubarray(nums []int) int {
	if len(nums) == 1 {
		return 1
	}
	var maxLength int = 1
	var upLength int = 1
	var downLength int = 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			upLength++
			downLength = 1
		}
		if nums[i] < nums[i-1] {
			downLength++
			upLength = 1
		}
		if nums[i] == nums[i-1] {
			downLength = 1
			upLength = 1
		}
		if downLength > maxLength {
			maxLength = downLength
		}
		if upLength > maxLength {
			maxLength = upLength
		}
	}
	return maxLength
}

//TIP <p>To run your code, right-click the code and select <b>Run</b>.</p> <p>Alternatively, click
// the <icon src="AllIcons.Actions.Execute"/> icon in the gutter and select the <b>Run</b> menu item from here.</p>

func main() {
	fmt.Print(longestMonotonicSubarray([]int{1, 4, 3, 3, 2}))
}
