window.addEventListener("DOMContentLoaded", (event) => {
  // Toggle the side navigation
  const sidebarToggle = document.body.querySelector("#sidebarToggle");
  if (sidebarToggle) {
    // Uncomment Below to persist sidebar toggle between refreshes
    // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
    //     document.body.classList.toggle('sb-sidenav-toggled');
    // }
    sidebarToggle.addEventListener("click", (event) => {
      event.preventDefault();
      document.body.classList.toggle("sb-sidenav-toggled");
      localStorage.setItem(
        "sb|sidebar-toggle",
        document.body.classList.contains("sb-sidenav-toggled")
      );
    });
  }
});

function validateForm() {
  const query1 = document.forms["input_form"]["query1"].value;
  const query2 = document.forms["input_form"]["query2"].value;
  const query3 = document.forms["input_form"]["query3"].value;
  const query4 = document.forms["input_form"]["query4"].value;
  const query5 = document.forms["input_form"]["query5"].value;
  const query6 = document.forms["input_form"]["query6"].value;

  if (
    query1 == "" ||
    query2 == "" ||
    query3 == "" ||
    query4 == "" ||
    query5 == "" ||
    query6 == ""
  ) {
    alert("Fields cannot be empty");
    return false;
  }
}
